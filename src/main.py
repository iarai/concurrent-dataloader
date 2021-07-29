from misc.time_helper import TimeHelper
from scratch_data_loader import ScratchDataLoader

IMAGENET_PATH_SCRATCH = "/scratch/imagenet"
IMAGENET_PATH_GLUSTER = "/scratch/imagenet"


class ActionPlayer:
    def __init__(self):
        pass

    def benchmark(self, action_name, action, repeat):
        for i in range(repeat):
            stopwatch.record(action_name + "_" + str(i))
            action()
            stopwatch.record(action_name + "_" + str(i))  
        stopwatch.get_results(action_name, imagenet_sdl.__len__())
        


if __name__ == "__main__":
    print("Hi scrach data loader!")
    imagenet_sdl = ScratchDataLoader(IMAGENET_PATH_SCRATCH, "val", 4, False, 1, False)
    stopwatch = TimeHelper()
    action_player = ActionPlayer()

    # ls (index) all images
    action_player.benchmark("indexing", imagenet_sdl.index_all, 50)

    # load random images
    action_player.benchmark("loading_random", imagenet_sdl.get_random_item, 10000)

    # save index to file
    action_player.benchmark("save_index", imagenet_sdl.save_index, 50)

    # load index from file
    action_player.benchmark("load_index", imagenet_sdl.load_index, 50)

    print("Recreating DataLoader")
    stopwatch.reset()
    imagenet_sdl = ScratchDataLoader(IMAGENET_PATH_GLUSTER, "train", 4, False, 1, False)
    imagenet_sdl.load_index()
    action_player.benchmark("loading_random", imagenet_sdl.get_random_item, 10000)
