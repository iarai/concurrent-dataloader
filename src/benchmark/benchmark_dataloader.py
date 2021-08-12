import torch
from action_player.action_player import ActionPlayer
from data_loader.async_data_loader import AsynchronousLoader
from dataset.scratch_dataset import ScratchDataset
from torch.utils.data import DataLoader

IMAGENET_PATH_SCRATCH = "/scratch/imagenet"


class BechmarkDataloader:
    def __init__(self, batch_size: int = 1, num_workers: int = 1, dataset: str = "val") -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers

        # todo extend to S3
        self.dataset = ScratchDataset(IMAGENET_PATH_SCRATCH, "val")
        self.dataset.load_index()

        # note: dataloader cannot be the part of this class due to pickling error
        # todo https://discuss.pytorch.org/t/torch-cuda-streams-stream-in-multiprocess-context-causes-error-cant-pickle-stream-objects/80625

    def create_dataloader(self, dataloader: str = "torch"):
        if dataloader == "torch":
            return DataLoader(
                dataset=self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                collate_fn=self.collate,
            )
        else:
            return AsynchronousLoader(
                data=self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                device=torch.device("cuda"),
                collate_fn=self.collate,
            )

    def collate(self, batch):
        imgs = [item for item in batch]
        return imgs

    def load_single(self, dataloader):
        try:
            n = next(iter(dataloader))
            print(len(n), len(n[0]))
        except StopIteration:
            pass

    def load_all(self, dataloader):
        # get single
        try:
            for i, batch in enumerate(dataloader):
                if i % 1000 == 0:
                    print(f"{len(batch)}, {i}")
            # print(len(n), len(n[0]))
        except (StopIteration, EOFError) as e:
            print(f"Exception raised: {str(e)}")


def benchmark_scratch_dataloader():
    action_player = ActionPlayer()
    bm = BechmarkDataloader(4, 2)
    dl = bm.create_dataloader("async")

    # todo calculate runtime for batches, and images per batch, try out different sizes of batchers and workers...

    print("Dataloader benchmark")
    # action_player.benchmark("loading_with_dataloader", lambda: bm.load_single(dl), 2)
    action_player.benchmark("loading_with_dataloader", lambda: bm.load_all(dl), 2)
    print("Done...")
