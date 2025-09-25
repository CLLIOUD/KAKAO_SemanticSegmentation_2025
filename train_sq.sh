CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29505 train.py \
 --result_dir "./pths/DDRNet001" \
 --epochs 20 \
 --lr 1.e-2 \
 --loadpath "./DDRNet23s_imagenet.pth" \
 --scale_range [0.75,1.25] \
 --crop_size [1024,1024] \
 --batch_size 16 \
 --dataset_dir "./SemanticDataset_final" \
 


