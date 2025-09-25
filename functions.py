from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import numpy as np
import ast
from glob import glob
import os

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def display_dataset_info(datadir, dataset):      
    print(f'Dataset path: {datadir}')    
    if dataset is not None:
        print(f"Found {len(dataset)} images.")    

def load_state_dict(model, state_dict):
    """
    model.module vs model key mismatch 문제를 자동으로 해결
    """
    from collections import OrderedDict

    new_state_dict = OrderedDict()

    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)

    for k, v in state_dict.items():
        if is_ddp:
            if not k.startswith('module.'):
                k = 'module.' + k
        else:
            if k.startswith('module.'):
                k = k[len('module.'):]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(new_state_dict.keys()) & model_keys

    total = len(model_keys)
    loaded = len(loaded_keys)
    percent = 100.0 * loaded / total if total > 0 else 0.0

    print(f"[Info] Loaded {loaded}/{total} state_dict entries ({percent:.2f}%) from checkpoint.")

class SegmentationTransform:
    def __init__(self, crop_size=[1024, 1024], scale_range=[0.5, 1.5]):
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.bilinear = transforms.InterpolationMode.BILINEAR
        self.nearest = transforms.InterpolationMode.NEAREST

    def __call__(self, image, label):
        # --- Random scale ---
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        width, height = image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        image = TF.resize(image, (new_height, new_width), interpolation=self.bilinear)
        label = TF.resize(label, (new_height, new_width), interpolation=self.nearest)

        # --- Pad if needed ---
        pad_h = max(self.crop_size[0] - new_height, 0)
        pad_w = max(self.crop_size[1] - new_width, 0)

        if pad_h > 0 or pad_w > 0:
            padding = (0, 0, pad_w, pad_h)  # left, top, right, bottom
            image = TF.pad(image, padding, fill=0)
            label = TF.pad(label, padding, fill=255)  # void class padding

        # --- Random crop ---
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
        image = TF.crop(image, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        # --- Random horizontal flip ---
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # --- To Tensor & Normalize ---
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)
        label = torch.from_numpy(np.array(label, dtype=np.uint8)).long()

        return image, label
    
class SegmentationDataset(Dataset):
    def __init__(self, root_dir, crop_size, subset, scale_range):
        self.crop_size = crop_size
        self.image_paths = sorted(glob(os.path.join(root_dir, "image", subset,"*", "*.*"), recursive=True))        
        self.label_paths = [self._get_label_path(p, root_dir) for p in self.image_paths]
        self.label_map = np.arange(256)
        self.transform = SegmentationTransform(crop_size, scale_range)

    def _get_label_path(self, image_path, root_dir):
        # image_path 예시: root_dir/image/train/xxx/yyy.png
        image_dir = os.path.join(root_dir, "image")
        label_dir = os.path.join(root_dir, "labelmap")

        # image/ 하위 상대 경로 구하기
        rel_path = os.path.relpath(image_path, image_dir)

        # 상대 경로를 폴더+파일명으로 분리
        rel_path_parts = rel_path.split(os.sep)
        file_name = rel_path_parts[-1]
        base_name, ext = os.path.splitext(file_name)

        # 파일명 변경
        if file_name.endswith("_leftImg8bit.png"):
            new_file_name = base_name.replace("_leftImg8bit", "_gtFine_CategoryId") + ".png"
        else:
            new_file_name = base_name + "_CategoryId.png"

        # 새 파일명으로 대체
        rel_path_parts[-1] = new_file_name

        # 최종 label path 구성: label_dir + 상대경로
        label_path = os.path.join(label_dir, *rel_path_parts)
        return label_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx]).convert("L")
        img, label = self.transform(img, label)
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(np.array(label, dtype=np.uint8))
        
        return img, label.long()

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label= 255, weight= None, aux_weights = [1, 0.4]):
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds, labels):
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        return self.criterion(preds, labels)

    def forward(self, preds, labels):
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)    
    
    
class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label= 255, weight = None, thresh = 0.6, aux_weights= [1, 0.4]):
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds, labels):
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels):
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)        

    
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_epochs=10, eta_min=0, last_epoch=-1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # linear warmup: from 0 to base_lr
            return [
                base_lr * float(self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + np.cos(np.pi * progress))
                for base_lr in self.base_lrs
            ]    
    


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, total_epochs=500, decay_epoch=1, power=0.9, last_epoch=-1) -> None:
        self.decay_epoch = decay_epoch
        self.total_epochs = total_epochs
        self.power = power
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch % self.decay_epoch != 0 or self.last_epoch > self.total_epochs:
            return self.base_lrs
        else:
            factor = (1 - self.last_epoch / float(self.total_epochs)) ** self.power
            return [factor*lr for lr in self.base_lrs]

class EpochWarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs=5, warmup_ratio=5e-4, warmup='linear',total_epochs=500, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        return [max(ratio * lr, 1e-7) for lr in self.base_lrs]

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_epochs:
            return self.get_warmup_ratio()
        return self.get_main_ratio()

    def get_warmup_ratio(self):
        alpha = self.last_epoch / self.warmup_epochs
        if self.warmup == 'linear':
            return self.warmup_ratio + (1. - self.warmup_ratio) * alpha
        else:
            return self.warmup_ratio ** (1. - alpha)

    def get_main_ratio(self):
        raise NotImplementedError
        
        
class WarmupPolyEpochLR(EpochWarmupLR):
    def __init__(self, optimizer, power=0.9, total_epochs=500, warmup_epochs=5, warmup_ratio=5e-4, warmup='linear', last_epoch=-1):
        self.power = power
        super().__init__(optimizer, warmup_epochs, warmup_ratio, warmup, total_epochs, last_epoch)

    def get_main_ratio(self):
        real_epoch = self.last_epoch - self.warmup_epochs
        real_total = self.total_epochs - self.warmup_epochs
        alpha = min(real_epoch / real_total, 1.0)
        return (1 - alpha) ** self.power