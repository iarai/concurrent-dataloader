from typing import List

import torch
from action_player.action_player import ActionPlayer
from data_loader.async_data_loader import AsynchronousLoader
from dataset.s3_clean_dataset import S3Dataset
from misc.time_helper import stopwatch
from torch.functional import Tensor
from torch_overrides.dataloader import DataLoader
from torch_overrides.worker import _worker_loop

IMAGENET_PATH_SCRATCH = "/scratch/imagenet"


@stopwatch("(2)-load_single")
def load_single(dataloader: DataLoader) -> None:
    try:
        _ = next(iter(dataloader))
    except (StopIteration, EOFError) as e:
        print(f"Exception raised: {str(e)}")


@stopwatch("(2)-load_all")
def load_all(dataloader: DataLoader) -> None:
    try:
        for i, batch in enumerate(dataloader):
            # if i % 10 == 0:
            # if isinstance(data, torch.utils.data.DataLoader)
            # print(f"{len(batch)}, {i}, {len(dataloader)}")
            pass
    except (StopIteration, EOFError, Exception) as e:
        print(f"Exception raised : {e}")


def collate(batch: List) -> Tensor:
    imgs = [item for item in batch]
    return imgs


@stopwatch("(1)-benchmark")
def benchmark_s3_dataloader(batch_size, num_workers, data_loader_type):
    action_player = ActionPlayer()

    _dataset = S3Dataset(mode="val", bucket_name="iarai-playground")
    _dataset.load_index()

    data_loader = None
    if data_loader_type == "async":
        data_loader = AsynchronousLoader(
            data=_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            device=torch.device("cuda"),
            collate_fn=collate)
    else:
        data_loader = DataLoader(
            dataset=_dataset,
            batch_size=8,
            num_workers=2,
            shuffle=False,
            collate_fn=collate,
            prefetch_factor=2,
        )

    # override the _worker_loop to inject @stopwatch
    torch.utils.data._utils.worker._worker_loop = _worker_loop

    print(f"Warmup ... batch {batch_size}, workers {num_workers}")
    action_player.benchmark("loading_with_dataloader", lambda: load_single(data_loader), 10, True)
    print("Warmup -- end")
    
    # real benchmark 
    action_player.benchmark("loading_with_dataloader", lambda: load_all(data_loader), 1, True)
