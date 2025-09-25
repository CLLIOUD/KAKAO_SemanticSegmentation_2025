import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from DDRNet import DDRNet
from functions import *
from pathlib import Path

def train(args):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank )
    device = torch.device("cuda", local_rank )


    # Dataset & Dataloader
    train_dataset = SegmentationDataset(args.dataset_dir, args.crop_size, 'train', args.scale_range)
    display_dataset_info(args.dataset_dir, train_dataset)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank ,drop_last=True,  shuffle=True)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                            num_workers=0, pin_memory=True)

    # Model
    print(f"[GPU {local_rank }] Before model setup")
    model = DDRNet(num_classes=args.num_classes).to(device)
    model = DDP(model, device_ids=[local_rank])
    print(f"[GPU {local_rank }] DDP initialized")

    # Loss, Optimizer, Scheduler
    criterion = CrossEntropy(ignore_label=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
#     scheduler = WarmupCosineAnnealingLR(optimizer, total_epochs=args.epochs, warmup_epochs=10, eta_min=1e-5)
    scheduler = WarmupPolyEpochLR(optimizer, total_epochs=args.epochs, warmup_epochs=5, warmup_ratio=5e-4)

    if args.loadpath is not None:
        map_location = {f'cuda:{0}': f'cuda:{local_rank }'}
        state_dict = torch.load(args.loadpath, map_location=map_location)
        load_state_dict(model, state_dict)

    if local_rank  == 0:
        os.makedirs(args.result_dir, exist_ok=True)
        log_path = os.path.join(args.result_dir, "log.txt")
        with open(log_path, 'w') as f:
            f.write("Epoch\t\tTrain-loss\t\tlearningRate\n")
    min_loss = 100.0
    for epoch in range(args.epochs):
        
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0.0
        loop = tqdm(dataloader, desc=f"[GPU {local_rank }] Epoch [{epoch+1}/{args.epochs}]", ncols=100) if local_rank  == 0 else dataloader

        for i, (imgs, labels) in enumerate(loop):
            optimizer.zero_grad(set_to_none=True)   
            
            imgs, labels = imgs.to(device), labels.to(device)            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize()
            
            total_loss += loss.item()
            if local_rank  == 0:
                loop.set_postfix(loss=loss.item(), avg_loss=total_loss/(i+1), lr=scheduler.get_last_lr()[0])

                
        torch.cuda.empty_cache()        
        dist.barrier()                 
        scheduler.step()

        if local_rank == 0 and ((total_loss / len(dataloader)) < min_loss):
            ckp_path = os.path.join(args.result_dir, f"model_best.pth")
            torch.save(model.state_dict(), ckp_path)
        
        if local_rank  == 0 and (epoch + 1) % 20 == 0:
            ckp_path = os.path.join(args.result_dir, f"model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckp_path)
            lr = scheduler.get_last_lr()
            lr = sum(lr) / len(lr)
            with open(log_path, "a") as f:
                f.write("\n%d\t\t%.4f\t\t%.8f" % (epoch + 1, total_loss / len(dataloader), scheduler.get_last_lr()[0]))                

    dist.destroy_process_group()


# ---------- Argparse ----------
if __name__ == "__main__":
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_P2P_DISABLE"] = "1" 
    os.environ["NCCL_IB_DISABLE"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str,  help="Path to dataset root", 
                        default="/home/user/ext_hdd/dataset/SeChal_2025/SemanticDataset_final_sampled" )
    parser.add_argument("--loadpath", type=str,  help="Path to dataset root", 
                        default=None )    # "ex: ./pths/DDRNet23s_imagenet.pth"
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--crop_size", default=[512, 1024], type=arg_as_list, help="crop size (H W)")    
    parser.add_argument("--scale_range", default=[0.75, 1.5], type=arg_as_list,  help="resize Input")    
    
    args = parser.parse_args()
    
    print(f'Initial learning rate: {args.lr}')
    print(f'Total epochs: {args.epochs}')
    print(f'dataset path: {args.dataset_dir}')
                  
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)               
    torch.multiprocessing.set_start_method('spawn', force=True)
    train(args)
