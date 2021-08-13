from typing import List

import torch
from action_player.action_player import ActionPlayer
from data_loader.async_data_loader import AsynchronousLoader
from dataset.scratch_dataset import ScratchDataset
from misc.time_helper import stopwatch
from torch.functional import Tensor
from torch_overrides.dataloader import DataLoader
from torch_overrides.worker import _worker_loop

IMAGENET_PATH_SCRATCH = "/scratch/imagenet"


class BechmarkDataloader:
    def __init__(self, batch_size: int = 1, num_workers: int = 1, dataset: str = "val") -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        limit = batch_size * num_workers * 10
        print(f"Data size: {limit}")

        # todo extend to S3
        self.dataset = ScratchDataset(IMAGENET_PATH_SCRATCH, "val", limit=limit)
        self.dataset.load_index()

        # note: dataloader cannot be the part of this class due to pickling error
        # todo https://discuss.pytorch.org/t/torch-cuda-streams-stream-in-multiprocess-context-causes-error-cant-pickle-stream-objects/80625 # flake8: noqa

    def create_dataloader(self, dataloader: str = "torch") -> DataLoader:
        if dataloader == "torch":
            return DataLoader(
                dataset=self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                collate_fn=self.collate,
                prefetch_factor=2,
            )
        else:
            return AsynchronousLoader(
                data=self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                device=torch.device("cuda"),
                collate_fn=self.collate,
                # prefetch_factor=2, 
            )

    def collate(self, batch: List) -> Tensor:
        imgs = [item for item in batch]
        return imgs

    def load_single(self, dataloader: DataLoader) -> None:
        try:
            _ = next(iter(dataloader))
        except (StopIteration, EOFError) as e:
            print(f"Exception raised: {str(e)}")

    def load_all(self, dataloader: DataLoader) -> None:
        try:
            for i, batch in enumerate(dataloader):
                # if i % 10 == 0:
                # if isinstance(data, torch.utils.data.DataLoader)
                # print(f"{len(batch)}, {i}, {len(dataloader)}")
                pass
        except (StopIteration, EOFError) as e:
            print(f"Exception raised: {str(e)}")


@stopwatch
def benchmark_scratch_dataloader():
    action_player = ActionPlayer()

    bm = BechmarkDataloader(batch_size=4, num_workers=2)
    # dl = bm.create_dataloader("async")
    dl = bm.create_dataloader("torch")

    # override the _worker_loop to inject @stopwatch
    torch.utils.data._utils.worker._worker_loop = _worker_loop

    print("Dataloader benchmark")
    action_player.benchmark("loading_with_dataloader", lambda: bm.load_single(dl), 1)
    action_player.benchmark("loading_with_dataloader", lambda: bm.load_all(dl), 1)
    print("Done...")
