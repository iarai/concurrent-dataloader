from misc.time_helper import TimeHelper
from s3_data_loader import S3DataLoader
from scratch_data_loader import ScratchDataLoader
import json

IMAGENET_PATH_SCRATCH = "/scratch/imagenet"
IMAGENET_PATH_GLUSTER = "/scratch/imagenet"


class ActionPlayer:
    def __init__(self):
        self.stopwatch = TimeHelper()

    def reset(self):
        self.stopwatch.reset()

    def benchmark(self, action_name, action, repeat):
        for i in range(repeat):
            self.stopwatch.record(action_name + "_" + str(i))
            action()
            self.stopwatch.record(action_name + "_" + str(i))
        self.stopwatch.get_results(action_name, repeat)


def test_data_loader(data_loader_instance, dataset="val"):
    action_player = ActionPlayer()

    # ls (index) all images
    action_player.benchmark("indexing", data_loader_instance.index_all, 50)

    # load random images
    action_player.benchmark(
        "loading_random", data_loader_instance.get_random_item, 10000
    )

    # save index to file
    action_player.benchmark("save_index", data_loader_instance.save_index, 50)

    # load index from file
    action_player.benchmark("load_index", data_loader_instance.load_index, 50)

    print("Recreating DataLoader")
    action_player.reset()
    data_loader_instance = ScratchDataLoader(IMAGENET_PATH_GLUSTER, dataset)
    data_loader_instance.load_index()
    action_player.benchmark(
        "loading_random", data_loader_instance.get_random_item, 10000
    )


def get_s3_cred():
    keys = None
    with open("s3_credentials.json", "r") as file:
        keys = json.load(file)
    return keys

if __name__ == "__main__":
    # test_scratch("val")
    # test_data_loader(ScratchDataLoader(IMAGENET_PATH_SCRATCH, "val"))
    keys = get_s3_cred()
    test_data_loader(
        S3DataLoader(
            mode="val",
            bucket_name="iarai-playground",
            access_key=keys["access_key"],
            secret_key=keys["secret"],
        )
    )
