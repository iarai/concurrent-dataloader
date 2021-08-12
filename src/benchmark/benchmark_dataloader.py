import torch
from action_player.action_player import ActionPlayer
from data_loader.async_data_loader import AsynchronousLoader
from dataset.scratch_dataset import ScratchDataset
from torch.utils.data import DataLoader

IMAGENET_PATH_SCRATCH = "/scratch/imagenet"


dataset = ScratchDataset(IMAGENET_PATH_SCRATCH, "val")
        # dataset.index_all()
        # dataset.save_index()
dataset.load_index()
batch_size = 10
num_workers = 4


def set_dataloader(self, dataloader: str = "async"):
    if dataloader == "async":
        use_async_dataloader()
    else:
        use_torch_dataloader()

def use_torch_dataloader():
    return DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False, 
        collate_fn=collate
    )

def use_async_dataloader():
    return AsynchronousLoader(
        data=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        device=torch.device("cuda"),
        collate_fn=collate,
    )

def collate(batch):
    imgs = [item for item in batch]
    return imgs

default_dataloader = use_async_dataloader()


def load_single():
    try:
        n = next(iter(default_dataloader))
        print(len(n), len(n[0]))
    except StopIteration:
        pass

def load_all():
    # get single
    try:
        for i, batch in enumerate(default_dataloader):
            if i % 1000 == 0:
                print(f"{len(batch)}, {i}")
        # print(len(n), len(n[0]))
    except (StopIteration, EOFError) as e:
        print(f"Exception raised: {str(e)}")

def benchmark_scratch_dataloader():
    action_player = ActionPlayer()
    #bm = BechmarkDataloader(10, 4)

    # todo calculate runtime for batches, and images per batch, try out different sizes of batchers and workers...
    # todo https://discuss.pytorch.org/t/torch-cuda-streams-stream-in-multiprocess-context-causes-error-cant-pickle-stream-objects/80625
    # todo, for some reason, this donesn't work as a class (the reason above)
    
    # action_player.benchmark("loading_with_dataloader", bm.load_single, 2)
    # action_player.benchmark("loading_with_dataloader", bm.load_all, 2)

    print("Dataloader benchmark")
    # action_player.benchmark("loading_with_dataloader", load_single, 2)
    action_player.benchmark("loading_with_dataloader", load_all, 2)
    print("Done...")
