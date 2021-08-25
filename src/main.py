import argparse
import logging

from action_player.mp_action_player import MPActionPlayer
from benchmark.benchmark_dataloader import benchmark_dataloader
from benchmark.benchmark_dataset import benchmark_s3_storage
from benchmark.benchmark_dataset import benchmark_scratch_storage
from benchmark.benchmark_local_image_dataset import benchmark_tensor_loading
from benchmark.benchmark_local_image_dataset import load_local_image_to_gpu
from benchmark.benchmark_local_image_dataset import load_random_local_image_to_gpu
from benchmark.benchmark_local_image_dataset import load_random_tensor_on_gpu
from benchmark.benchmark_local_image_dataset import load_random_tensor_to_gpu
from misc.random_generator import RandomGenerator


def handle_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--action",
        help="An option to benchmark (s3, scratch, random_gpu, random_to_gpu, random_image)",
        default="random_gpu",
    )
    parser.add_argument("-m", "--dataset", help="Default dataset (val or train)", default="val")
    parser.add_argument("-args", "--args", nargs="+", help="Additional arguments")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    action_repeat = 16
    verbose = True

    # interpret arguments
    parser = handle_arguments()
    args = parser.parse_args()

    # load dataset
    dataset = args.dataset

    if args.action == "s3":
        benchmark_s3_storage(dataset, mp=True)
    elif args.action == "scratch":
        benchmark_scratch_storage(dataset)
    elif args.action == "random_gpu":
        benchmark_tensor_loading(
            load_random_tensor_on_gpu, warmup_cycle=False, action_repeat=action_repeat, verbose=verbose
        )
        benchmark_tensor_loading(
            load_random_tensor_on_gpu, warmup_cycle=True, action_repeat=action_repeat, verbose=verbose
        )
    elif args.action == "random_to_gpu":
        benchmark_tensor_loading(
            load_random_tensor_to_gpu, warmup_cycle=False, action_repeat=action_repeat, verbose=verbose
        )
        benchmark_tensor_loading(
            load_random_tensor_to_gpu, warmup_cycle=True, action_repeat=action_repeat, verbose=verbose
        )
    elif args.action == "single_image":
        benchmark_tensor_loading(
            load_local_image_to_gpu, warmup_cycle=False, action_repeat=action_repeat, verbose=verbose
        )
        benchmark_tensor_loading(
            load_local_image_to_gpu, warmup_cycle=True, action_repeat=action_repeat, verbose=verbose
        )
    elif args.action == "random_image":
        benchmark_tensor_loading(
            load_random_local_image_to_gpu, warmup_cycle=False, action_repeat=action_repeat, verbose=verbose
        )
        benchmark_tensor_loading(
            load_random_local_image_to_gpu, warmup_cycle=True, action_repeat=action_repeat, verbose=verbose
        )
    elif args.action == "mp":
        benchmark_tensor_loading(load_local_image_to_gpu, True, action_repeat, verbose=verbose)
        rng = RandomGenerator()
        mpap = MPActionPlayer(rng, num_workers=8, pool_size=4)
        benchmark_tensor_loading(load_local_image_to_gpu, True, action_repeat, mpap, verbose=verbose)
        benchmark_tensor_loading(load_random_tensor_to_gpu, True, action_repeat, mpap, verbose=verbose)
        benchmark_tensor_loading(load_random_tensor_on_gpu, True, action_repeat, mpap, verbose=verbose)
    elif args.action == "dataloader":
        batch_size = int(args.args[0])
        num_workers = int(args.args[1])
        data_loader_type = args.args[2]
        benchmark_dataloader(batch_size, num_workers, data_loader_type)
    else:
        parser.print_help()
