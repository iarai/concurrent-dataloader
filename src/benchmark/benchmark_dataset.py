import json
import logging
from typing import Type

from dataset.s3_dataset import S3Dataset
from dataset.scratch_dataset import ScratchDataset
from action_player.action_player import ActionPlayer

IMAGENET_PATH_SCRATCH = "/scratch/imagenet"


# main function that defines the testing order ... e.g. index, load, save
def benchmark_data_loader(data_loader_instance: Type[S3Dataset], skip_indexing: bool = False) -> None:
    action_player = ActionPlayer()

    # ls (index) all images
    if not skip_indexing:
        action_player.benchmark("indexing", data_loader_instance.index_all, 50)
    else:
        action_player.benchmark("load_index", data_loader_instance.load_index, 50)
        action_player.benchmark("loading_random", data_loader_instance.get_random_item, 10000)
        return

    # load random images
    action_player.benchmark("loading_random", data_loader_instance.get_random_item, 10000)

    # save index to file
    action_player.benchmark("save_index", data_loader_instance.save_index, 50)

    # load index from file
    action_player.benchmark("load_index", data_loader_instance.load_index, 50)


def benchmark_scratch_storage(dataset: str = "val") -> None:
    logging.info("Starting benchmark ... Using scratch")
    # test dataloader with scratch
    benchmark_data_loader(ScratchDataset(IMAGENET_PATH_SCRATCH, dataset))


def benchmark_s3_storage(dataset: str = "val") -> None:
    logging.info("Starting benchmark ... Using S3")
    # read s3 credentials
    keys = get_s3_cred()
    # test dataloader with scratch
    benchmark_data_loader(
        S3Dataset(
            mode=dataset, bucket_name="iarai-playground", access_key=keys["access_key"], secret_key=keys["secret"],
        ),
        skip_indexing=False,
    )


# load credentials for S3
def get_s3_cred() -> None:
    __keys = None
    with open("s3_credentials.json", "r") as file:
        __keys = json.load(file)
    return __keys
