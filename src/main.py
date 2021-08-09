#! /usr/bin/python3
import argparse
import logging 

from benchmark.data_loader import benchmark_scratch_storage
from benchmark.data_loader import benchmark_s3_storage
from benchmark.image_loader import benchmark_tensor_loading
from benchmark.image_loader import load_random_image_to_gpu
from benchmark.image_loader import load_random_tensor_on_gpu
from benchmark.image_loader import load_random_tensor_to_gpu
from misc.mp_action_player import MPActionPlayer


def handle_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--action",
        help="An option to benchmark (s3, scratch, random_gpu, random_to_gpu, random_image)",
        default="random_gpu",
    )
    parser.add_argument("-m", "--dataset", help="Default dataset (val or train)", default="val")
    return parser


if __name__ == "__main__":
    # interpret arguments
    parser = handle_arguments()
    args = parser.parse_args()

    # load dataset
    dataset = args.dataset

    if args.action == "s3":
        benchmark_s3_storage(dataset)
    elif args.action == "scratch":
        benchmark_scratch_storage(dataset)
    elif args.action == "random_gpu":
        benchmark_tensor_loading(load_random_tensor_on_gpu, warmup_cycle=False, action_repeat=200)
        benchmark_tensor_loading(load_random_tensor_on_gpu, warmup_cycle=True, action_repeat=200)
    elif args.action == "random_to_gpu":
        benchmark_tensor_loading(load_random_tensor_to_gpu, warmup_cycle=False, action_repeat=200)
        benchmark_tensor_loading(load_random_tensor_to_gpu, warmup_cycle=True, action_repeat=200)
    elif args.action == "random_image":
        benchmark_tensor_loading(load_random_image_to_gpu, warmup_cycle=False, action_repeat=200)
        benchmark_tensor_loading(load_random_image_to_gpu, warmup_cycle=True, action_repeat=200)
    else:
        parser.print_help()

    # test_tensor_loading(load_image, True, 200)
    # action_player = MPActionPlayer(num_workers=8, pool_size=4)
    # test_tensor_loading(load_image, True, 200, action_player)
