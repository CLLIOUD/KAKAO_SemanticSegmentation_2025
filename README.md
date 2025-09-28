 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/README.md b/README.md
index eefd93faeb962162caf8d0353cfc34ff980e0a3b..098e854a22a552f574aa2217fc0d3a9c9741e62f 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,145 @@
-# KAKAO_SemanticSegmentation_2025
+# KAKAO Semantic Segmentation 2025
+
+This repository provides the training and inference pipeline used for the KAKAO 2025 semantic segmentation task. It fine-tunes a DDRNet-based dual-resolution encoder on road-scene imagery and delivers end-to-end scripts covering training, evaluation, prediction export, and computational profiling.
+
+> 🇰🇷 **요약**: 이 저장소는 KAKAO 2025 도로 장면 시맨틱 세그멘테이션 과제를 위한 학습·추론 파이프라인을 담고 있습니다. DDRNet 기반의 이중 해상도 인코더를 미세 조정하며, 학습/평가/추론/프로파일링 전체 과정을 수행하는 스크립트를 제공합니다.
+
+## Repository map
+
+| File | Purpose |
+| --- | --- |
+| `train.py` | Distributed training entrypoint. Builds the dataset, model, optimizer, and learning-rate schedule, then launches a multi-GPU training loop using PyTorch DDP. |
+| `train_sq.sh` | Convenience shell script that launches `train.py` with the hyperparameters used for the baseline submission. |
+| `DDRNet.py` | Implementation of DDRNet, including residual blocks, the dense attention pyramid pooling module (DAPPM), and segmentation heads. |
+| `functions.py` | Dataset utilities, data augmentations, loss functions, and learning-rate schedulers shared by training scripts. |
+| `prediction.py` | Loads trained weights and exports semantic labels and color-mapped overlays for the test set. |
+| `evaluation.py` | Computes pixel accuracy statistics (per-class IoU and mIoU) given predicted label maps and ground-truth annotations. |
+| `computation_time.py` | Measures FLOPs, parameter count, memory usage, and inference latency for DDRNet using THOP. |
+| `DDRNet23s_imagenet.pth` | Pre-trained ImageNet checkpoint used to initialize DDRNet. |
+
+| 파일 | 용도 |
+| --- | --- |
+| `train.py` | PyTorch DDP를 활용해 멀티 GPU 학습 루프를 수행하는 분산 학습 진입점입니다. 데이터셋/모델/옵티마이저/러닝레이트 스케줄을 구성합니다. |
+| `train_sq.sh` | 기준 실험에서 사용한 하이퍼파라미터로 `train.py`를 실행하는 편의 스크립트입니다. |
+| `DDRNet.py` | 잔차 블록, DAPPM, 시맨틱 헤드를 포함한 DDRNet 구현입니다. |
+| `functions.py` | 데이터셋 유틸, 데이터 증강, 손실 함수, 학습률 스케줄러 등을 제공합니다. |
+| `prediction.py` | 학습된 가중치를 불러와 테스트 세트에 대한 라벨 맵과 컬러 오버레이를 생성합니다. |
+| `evaluation.py` | 예측과 정답을 비교해 클래스별 IoU와 mIoU를 계산합니다. |
+| `computation_time.py` | THOP을 이용해 FLOPs/파라미터 수를 측정하고, 메모리 사용량과 추론 지연 시간을 분석합니다. |
+| `DDRNet23s_imagenet.pth` | DDRNet 초기화를 위한 ImageNet 사전 학습 가중치입니다. |
+
+## Data preparation
+
+The pipeline expects the dataset directory to contain:
+
+```
+<dataset_root>/
+  image/
+    train/<scene_id>/<frame>.png
+    val/<scene_id>/<frame>.png
+    test/<scene_id>/<frame>.png
+  labelmap/
+    train/<scene_id>/<frame>_gtFine_CategoryId.png
+    val/<scene_id>/<frame>_gtFine_CategoryId.png
+```
+
+* `train_sq.sh` assumes the dataset is unpacked into `./SemanticDataset_final`.
+* Test data lacks labels. Predicted labels are saved under `result/label/...` with the same relative paths as the inputs.
+
+> 🇰🇷 **데이터 준비**
+>
+> * 데이터셋은 `image/{train,val,test}`와 `labelmap/{train,val}` 구조를 따라야 합니다.
+> * `train_sq.sh`는 데이터가 `./SemanticDataset_final` 경로에 압축 해제되어 있다고 가정합니다.
+> * 테스트 데이터에는 라벨이 없으므로, 예측 라벨은 입력과 동일한 경로 구조를 갖는 `result/label/...`에 저장됩니다.
+
+## Data pipeline (`functions.py`)
+
+* `SegmentationDataset` searches under `image/<split>` and automatically maps each image file to the corresponding label file under `labelmap/`. The helper `_get_label_path` replaces the `_leftImg8bit` suffix with `_gtFine_CategoryId` so the correct annotation is found.
+* `SegmentationTransform` performs augmentation: random scaling, padding, random cropping to the configured `crop_size`, horizontal flipping, and ImageNet-style normalization. Labels are padded with `255` (ignore index) and converted to tensors.
+* `CrossEntropy` and `OhemCrossEntropy` wrap `nn.CrossEntropyLoss` to support auxiliary heads (DDRNet produces both a main and auxiliary prediction during training).
+* `WarmupPolyEpochLR` and `WarmupCosineAnnealingLR` provide warmup + decay schedules that adjust the optimizer learning rate after each epoch.
+* `load_state_dict` handles loading checkpoints regardless of whether they were saved from a DDP model (`module.` prefix) or a single-GPU model.
+
+> 🇰🇷 **데이터 파이프라인 개요**
+>
+> * `SegmentationDataset`은 `image/<split>`의 이미지를 순회하며 파일명 규칙을 이용해 라벨 경로를 자동 매핑합니다.
+> * `SegmentationTransform`는 랜덤 스케일, 패딩, 랜덤 크롭, 좌우 반전, 정규화를 적용해 학습 데이터를 증강합니다. 라벨은 `255`(ignore index)로 패딩됩니다.
+> * `CrossEntropy` / `OhemCrossEntropy`는 보조 헤드를 포함한 손실 계산을 지원합니다.
+> * `WarmupPolyEpochLR`, `WarmupCosineAnnealingLR`는 워밍업 후 다항/코사인 감쇠 학습률 스케줄을 제공합니다.
+> * `load_state_dict`는 DDP/단일 GPU 체크포인트의 접두사 차이를 자동으로 보정합니다.
+
+## Model architecture (`DDRNet.py`)
+
+DDRNet employs a dual-path encoder:
+
+1. **Low-resolution branch**: stacks residual `BasicBlock` layers with downsampling to extract high-level semantics. `compression3/4` project features for cross-path fusion.
+2. **High-resolution branch**: maintains spatial detail via parallel residual layers (`layer3_`, `layer4_`, `layer5_`). Down-sampling modules (`down3`, `down4`) pass detail back to the low-res branch.
+3. **DAPPM**: multi-scale pooling module that aggregates global context and merges it with the high-resolution stream.
+4. **Segmentation heads**: `segmenthead` upsamples logits back to the input resolution. During training an auxiliary head (`seghead_extra`) supervises intermediate features.
+
+The forward pass fuses features between both resolutions, applies DAPPM to the lowest-resolution tensor, and returns either `(main_logits, aux_logits)` in training mode or `main_logits` during inference.
+
+> 🇰🇷 **모델 아키텍처 요약**
+>
+> 1. 저해상도 분기: 다운샘플링된 잔차 블록으로 고수준 의미 정보를 추출하며, `compression3/4` 모듈이 분기 간 피처를 교환합니다.
+> 2. 고해상도 분기: `layer3_`, `layer4_`, `layer5_`로 공간 정보를 유지하고, `down3/4` 모듈이 세부 정보를 저해상도 분기로 전달합니다.
+> 3. DAPPM: 다중 스케일 풀링 출력을 업샘플링 후 결합해 전역 문맥을 강화합니다.
+> 4. 세그멘테이션 헤드: 보간을 통해 입력 해상도로 복원하며, 학습 시 보조 헤드(`seghead_extra`)도 손실에 참여합니다.
+
+## Training loop (`train.py`)
+
+* Initializes NCCL distributed communication and sets the current CUDA device based on `LOCAL_RANK`.
+* Builds a `SegmentationDataset` and wraps it with a `DistributedSampler` for even sharding across GPUs. DataLoader workers are pinned for faster host-to-device transfers.
+* Constructs the DDRNet model and wraps it with `DistributedDataParallel` (`DDP`). Optionally loads ImageNet weights through `load_state_dict`.
+* Uses stochastic gradient descent with momentum and weight decay. The learning rate follows `WarmupPolyEpochLR`: a 5-epoch warmup followed by polynomial decay to 0.
+* Logs epoch losses and saves checkpoints under `result_dir`. The best checkpoint (lowest training loss) and periodic snapshots (`model_epoch{N}.pth`) are persisted by rank 0 only.
+* `train_sq.sh` launches the training script via `torchrun`, targeting 4 GPUs (IDs 0–3) with batch size 16, crop size 1024×1024, scale jitter `[0.75, 1.25]`, and 20 epochs.
+
+> 🇰🇷 **학습 루프 요약**
+>
+> * NCCL 기반 분산 통신을 초기화하고 `LOCAL_RANK`에 맞춰 GPU를 할당합니다.
+> * `DistributedSampler`로 데이터를 GPU별로 균등 분배하며, 고정된 worker를 사용해 로딩 속도를 높입니다.
+> * DDRNet을 구성해 `DistributedDataParallel`로 래핑하고, 필요 시 ImageNet 가중치를 로딩합니다.
+> * 모멘텀/가중 감쇠가 있는 SGD와 워밍업+다항 감쇠 스케줄(`WarmupPolyEpochLR`)을 사용합니다.
+> * 에폭 손실을 로깅하고 체크포인트(`result_dir`)를 저장합니다. 최상위 모델과 주기적 스냅샷은 0번 랭크만 기록합니다.
+> * `train_sq.sh`는 4개 GPU(0–3), 배치 크기 16, 크롭 1024×1024, 스케일 지터 `[0.75, 1.25]`, 총 20 에폭으로 `torchrun`을 실행합니다.
+
+## Inference (`prediction.py`)
+
+* `TestSegmentationDataset` iterates over test images and returns tensors without augmentations.
+* `load_model` wraps DDRNet in `DataParallel`, loads checkpoint weights (handling prefix mismatches), moves the model to the selected device, and switches to eval mode.
+* During prediction, each image is forwarded individually. Softmax+argmax produces the final class label per pixel.
+* `save_prediction` writes out the grayscale label map plus a Turbo colormap overlay for visual inspection. Output directory structure mirrors the input layout.
+
+> 🇰🇷 **추론 파이프라인**
+>
+> * `TestSegmentationDataset`은 증강 없이 테스트 이미지를 로드합니다.
+> * `load_model`이 DDRNet을 `DataParallel`로 감싼 뒤 체크포인트를 로드하고 평가 모드로 전환합니다.
+> * 각 이미지를 순차적으로 추론하고 Softmax+Argmax로 픽셀별 클래스를 결정합니다.
+> * `save_prediction`은 회색조 라벨과 Turbo 컬러맵 시각화를 입력과 동일한 구조로 저장합니다.
+
+## Evaluation (`evaluation.py`)
+
+Given a directory of predicted label maps and the ground-truth label directory, `evaluate` aligns files by ID, ignores pixels with value 255, accumulates a confusion matrix, and reports class-wise IoU along with mean IoU (mIoU).
+
+> 🇰🇷 **평가 지표**
+>
+> 예측 라벨과 정답 라벨을 ID 기준으로 맞춰 255(ignore)를 제외하고 혼동 행렬을 누적한 뒤 클래스별 IoU와 mIoU를 계산합니다.
+
+## Computational profiling (`computation_time.py`)
+
+Measures several deployment metrics on a CUDA device:
+
+* Parameter count and FLOPs (via THOP).
+* Peak GPU memory consumption for a 1080×1920 input.
+* Forward-pass latency statistics over 9 repeated groups of 200 iterations (after warmup), reporting both median latency and frames per second (FPS).
+
+These scripts together allow you to train, validate, test, and characterize the DDRNet segmentation model end-to-end for the KAKAO 2025 semantic segmentation challenge.
+
+> 🇰🇷 **성능/복잡도 분석**
+>
+> * THOP으로 파라미터 수와 FLOPs를 산출합니다.
+> * 1080×1920 입력 기준 GPU 메모리 사용량을 측정합니다.
+> * 200회 반복을 9그룹으로 수행한 추론 지연 시간과 FPS를 제공합니다.
+
+이 문서에 정리된 스크립트들을 활용하면 DDRNet 기반 모델의 학습부터 추론, 평가, 성능 분석까지 한 번에 수행할 수 있습니다.
 
