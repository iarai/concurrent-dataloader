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

def collate(batch: List) -> Tensor:
    imgs = [item for item in batch]
    return imgs


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


@stopwatch("(1)-benchmark")
def benchmark_scratch_dataloader(args):
    action_player = ActionPlayer()

    # bm = BechmarkDataloader(batch_size=int(args[0]), num_workers=int(args[1]), limit=False)


    _dataset = S3Dataset(mode="val", bucket_name="iarai-playground")
    _dataset.load_index()
    # print(_dataset.__getitem__(1))
    # print(_dataset.__len__())

    dl = None
    if args[2] == "async":
        dl = AsynchronousLoader(
            data=_dataset,
            batch_size=int(args[0]),
            num_workers=int(args[1]),
            shuffle=False,
            device=torch.device("cuda"),
            collate_fn=collate)
    else:
        dl = DataLoader(
                dataset=_dataset,
                batch_size=8,
                num_workers=2,
                shuffle=False,
                collate_fn=collate,
                prefetch_factor=2,
            )

    # override the _worker_loop to inject @stopwatch
    torch.utils.data._utils.worker._worker_loop = _worker_loop

    print(f"Warmup ... batch {args[0]}, workers {args[1]}")
    action_player.benchmark("loading_with_dataloader", lambda: load_single(dl), 10, True)
    print("Warmup -- end")
    action_player.benchmark("loading_with_dataloader", lambda: load_all(dl), 1, True)
