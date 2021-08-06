#! /usr/bin/python3
import argparse

from benchmark.data_loader import test_s3
from benchmark.data_loader import test_scratch
from benchmark.image_loader import load_image
from benchmark.image_loader import load_random_on_gpu
from benchmark.image_loader import load_random_to_gpu
from benchmark.image_loader import test_tensor_loading
from misc.mp_action_player import MPActionPlayer


def handle_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--benchmark", help="An option to benchmark (s3, scratch, random_gpu, random_to_gpu, random_image)", default="random_gpu")
    parser.add_argument("-m", "--dataset", help="Default dataset (val or train)", default="val")
    return parser


if __name__ == "__main__":
    # interpret arguments
    parser = handle_arguments()
    args = parser.parse_args()

    # load dataset
    dataset = args.dataset

    if args.benchmark == "s3":
        test_s3(dataset)
    elif args.benchmark == "scratch":
        test_scratch(dataset)
    elif args.benchmark == "random_gpu":
        test_tensor_loading(load_random_on_gpu, False, 200)
        test_tensor_loading(load_random_on_gpu, True, 200)
    elif args.benchmark == "random_to_gpu":
        test_tensor_loading(load_random_to_gpu, False, 200)
        test_tensor_loading(load_random_to_gpu, True, 200)
    elif args.benchmark == "random_image":
        test_tensor_loading(load_image, False, 200)
        test_tensor_loading(load_image, True, 200)
    else:
        parser.print_help()

    test_tensor_loading(load_image, True, 200)
    action_player = MPActionPlayer(num_workers=8, pool_size=4)
    test_tensor_loading(load_image, True, 200, action_player)
