import json

from misc.action_player import ActionPlayer

from data_loader.s3_data_loader import S3DataLoader
from data_loader.scratch_data_loader import ScratchDataLoader

IMAGENET_PATH_SCRATCH = "/scratch/imagenet"


def test_data_loader(data_loader_instance, skip_indexing=False):
    action_player = ActionPlayer()

    # ls (index) all images
    if not skip_indexing:
        action_player.benchmark("indexing", data_loader_instance.index_all, 50)
    else:
        action_player.benchmark("load_index", data_loader_instance.load_index, 50)
        action_player.benchmark(
            "loading_random", data_loader_instance.get_random_item, 10000
        )
        return

    # load random images
    action_player.benchmark(
        "loading_random", data_loader_instance.get_random_item, 10000
    )

    # save index to file
    action_player.benchmark("save_index", data_loader_instance.save_index, 50)

    # load index from file
    action_player.benchmark("load_index", data_loader_instance.load_index, 50)


def test_scratch(dataset="val"):
    # test dataloader with scratch
    test_data_loader(ScratchDataLoader(IMAGENET_PATH_SCRATCH, dataset))


def test_s3(dataset="val"):
    # read s3 credentials
    keys = get_s3_cred()
    # test dataloader with scratch
    test_data_loader(
        S3DataLoader(
            mode=dataset,
            bucket_name="iarai-playground",
            access_key=keys["access_key"],
            secret_key=keys["secret"],
        ),
        True,
    )


def get_s3_cred():
    keys = None
    with open("s3_credentials.json", "r") as file:
        keys = json.load(file)
    return keys
