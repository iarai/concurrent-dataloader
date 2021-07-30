import argparse
import json

import torch

from misc.time_helper import TimeHelper
from s3_data_loader import S3DataLoader
from scratch_data_loader import ScratchDataLoader

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


def test_data_loader(data_loader_instance, skip_indexing=False):
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
        True
    )

def get_s3_cred():
    keys = None
    with open("s3_credentials.json", "r") as file:
        keys = json.load(file)
    return keys

def load_random_on_gpu():
    # creates a tensor directly on a GPU (same size as ImageNet)
    torch.rand(469, 387, device=torch.device('cuda:0'))

def load_random_to_gpu():
    # creates a tensor directly on a GPU (same size as ImageNet)
    # check: 
    #   https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
    #   https://pytorch.org/docs/stable/notes/cuda.html
    torch.rand(469, 387).cuda()

def test_random_tensor(create_tensor_fn, warmup_cycle=False, repeat=10):
    action_player = ActionPlayer()  
    # warmup cycle
    action_name = "load_tensor"
    if warmup_cycle:
        for i in range(30):
            torch.rand(256, 256, device=torch.device('cuda:0'))
        action_name = action_name + "_with_warmup"
    action_player.benchmark(action_name, create_tensor_fn, repeat)

def handle_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--benchmark",
                        help="An option to benchmark (s3, scratch, random_gpu, random_to_gpu)",
                        default="random_gpu")
    parser.add_argument("-m", "--dataset",
                    help="Default dataset (val or train)",
                    default="val")
    return parser.parse_args()

if __name__ == "__main__":
    # interpret arguments
    args = handle_arguments()
    
    # load dataset
    dataset = args.dataset

    if args.benchmark == "s3":
        test_s3(dataset)
    elif args.benchmark == "scratch":
        test_scratch(dataset)
    elif args.benchmark == "random_gpu":
        test_random_tensor(load_random_on_gpu, False, 200)
        test_random_tensor(load_random_on_gpu, True, 200)
    elif args.benchmark == "random_to_gpu":
        test_random_tensor(load_random_to_gpu, False, 200)
        test_random_tensor(load_random_to_gpu, True, 200)
