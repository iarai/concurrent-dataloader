import logging
from typing import Type

from action_player.action_player import ActionPlayer
from action_player.mp_action_player import MPActionPlayer
from dataset.s3_dataset import S3Dataset
from dataset.scratch_dataset import ScratchDataset

IMAGENET_PATH_SCRATCH = "/scratch/imagenet"


# main function that defines the testing order ... e.g. index, load, save
def benchmark_data_loader(data_loader_instance: Type[S3Dataset], skip_indexing: bool = False, mp: bool = False) -> None:
    action_player = None
    if not mp:
        action_player = ActionPlayer()
    else:
        assert skip_indexing, "Indexing cannot be performed by Multi-Processing ActionPlayer"
        action_player = MPActionPlayer()

    # ls (index) all images
    if not skip_indexing:
        logging.info("Indexing")
        action_player.benchmark("indexing", data_loader_instance.index_all, 5)
    else:
        logging.info("Loading")
        action_player.benchmark("load_index", data_loader_instance.load_index, 5)
        action_player.benchmark("loading_random", data_loader_instance.get_random_item, 10000)
        return

    # load random images
    action_player.benchmark("loading_random", data_loader_instance.get_random_item, 5)

    # load index from file
    logging.info("Loading index... ")
    data_loader_instance.load_index()

    action_player.benchmark("load_index", data_loader_instance.load_index, 2)
    logging.info(f"Loading index... Done. Len {data_loader_instance.__len__()}")

    data_loader_instance.get_random_item()
    action_player.benchmark("loading_random", data_loader_instance.get_random_item, 2)


def benchmark_scratch_storage(dataset: str = "val", mp: bool = False) -> None:
    logging.info("Starting benchmark ... Using scratch")
    # test dataloader with scratch
    benchmark_data_loader(ScratchDataset(IMAGENET_PATH_SCRATCH, dataset), mp=mp)


def benchmark_s3_storage(dataset: str = "val", mp: bool = False) -> None:
    logging.info("Starting benchmark ... Using S3")
    # test dataloader with scratch
    benchmark_data_loader(
        S3Dataset(mode=dataset, bucket_name="iarai-playground",), skip_indexing=True, mp=mp,
    )
