from torch.utils.data import DataLoader
from data_loader.async_data_loader import AsynchronousLoader
from dataset.scratch_dataset import ScratchDataset
import torch

IMAGENET_PATH_SCRATCH = "/scratch/imagenet"

def get_scratch_dataloader():
    sd = ScratchDataset(IMAGENET_PATH_SCRATCH, "val")
    sd.index_all()
    __scratch_dataloader = DataLoader(dataset=sd,
    batch_size=2,
    num_workers=4,
    shuffle=False,
    collate_fn=my_collate_fn)
    return __scratch_dataloader

def get_scratch_async_dataloader():
    sd = ScratchDataset(IMAGENET_PATH_SCRATCH, "val")
    sd.index_all()
    __scratch_dataloader = AsynchronousLoader(data=sd,
    batch_size=2,
    num_workers=2,
    shuffle=False,
    device=torch.device("cuda"),
    collate_fn=my_collate_fn)
    return __scratch_dataloader

def my_collate_fn(batch):
    imgs = [item for item in batch]
    return imgs

def benchmark_scratch_dataloader():
    r = next(iter(get_scratch_dataloader()))   
    print(f"Return from torch dataloader: {type(r)}")
    for l in r:
        print(f"{type(l)}")
    
    r = next(iter(get_scratch_async_dataloader()))   
    print(f"Return from async_dataloader: {type(r)}")
    for l in r:
        print(f"{type(l)}")