import argparse
import logging
import sys
from functools import partial
from pathlib import Path
from typing import Callable
from typing import Type

import torch
from action_player.action_player import ActionPlayer
from action_player.mp_action_player import MPActionPlayer
from main import init_benchmarking
from misc.random_generator import RandomGenerator
from PIL import Image
from torchvision import transforms

transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),])
rng = RandomGenerator()

IMAGE_PATH = "resources/"


def load_random_local_image_to_gpu(device: str = "cuda:0") -> torch.Tensor:
    # get all images and choose one at random
    image_path_list = list(Path(IMAGE_PATH).glob("*.JPEG"))
    num = rng.get_int(0, len(image_path_list) - 1)
    img_to_load = image_path_list[num]
    logging.debug(f"Oppening local random image: {img_to_load}, rn: {num}")
    image = Image.open(img_to_load)
    # perform transforms and send to GPU
    image_tensor = transforms(image).cuda(device)
    return image_tensor


def load_local_image_to_gpu(device: str = "cuda:0") -> torch.Tensor:
    image_path_list = list(Path(IMAGE_PATH).glob("*.JPEG"))
    img_to_load = image_path_list[0]
    logging.debug(f"Oppening local image: {img_to_load}")
    image = Image.open(img_to_load)
    # perform transforms and send to GPU
    image_tensor = transforms(image).cuda(device)
    return image_tensor


def load_random_tensor_on_gpu(device: str = "cuda:0") -> None:
    # creates a tensor directly on a GPU (same size as ImageNet)
    # check:
    #   https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
    #   https://pytorch.org/docs/stable/notes/cuda.html
    torch.rand(469, 387, device=torch.device(device))


def load_random_tensor_to_gpu(device: str = "cuda:0") -> None:
    # Returns a copy of this object in CUDA memory. If this object is already in CUDA memory and on the correct device,
    # then no copy is performed and the original object is returned.
    # However, this first creates CPU tensor, and THEN transfers it to GPU… this is really slow.
    # Instead, create the tensor directly on the device you want.

    # check:
    #   https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
    #   https://pytorch.org/docs/stable/generated/torch.Tensor.cuda.html
    #   https://pytorch.org/docs/stable/notes/cuda.html
    torch.rand(469, 387).cuda(device)


def benchmark_tensor_loading(
    create_tensor_fn: Callable,
    output_base_folder: Path,
    warmup_cycle: bool = False,
    action_repeat: int = 10,
    action_player: Type[ActionPlayer] = None,
    device: str = "cuda:0",
) -> None:
    if action_player is None:
        action_player = ActionPlayer()

    # warmup cycle
    logging.info(f"Benchmarking tensor loading, using warmup cycle {warmup_cycle}")
    action_name = create_tensor_fn.__name__

    # before performing the benchmark, perform a warmup cycle, usually helping GPU to speed up later processing
    # to avoid (hidden/unknown) "one-time" startup costs
    if warmup_cycle:
        for _ in range(30):
            torch.rand(256, 256, device=torch.device(device))
        action_name = action_name + "_with_warmup"
    action_player.benchmark(
        action_name=action_name,
        action=partial(create_tensor_fn, device=device),
        repeat=action_repeat,
        output_base_folder=output_base_folder,
    )


def handle_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--action",
        help="An option to benchmark (s3, scratch, random_gpu, random_to_gpu, random_image)",
        default="random_gpu",
    )
    parser.add_argument("--output_base_folder", type=Path, default=Path("benchmark_output"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--action_repeat", type=int, default=16)

    return parser


# TODO we should not use data from warmup cycle!!!
def main(*args):
    parser = handle_arguments()
    args = parser.parse_args(args)
    device = args.device
    action_repeat = args.action_repeat
    output_base_folder = init_benchmarking(args, action="_".join(["benchmark_tensor_loading", args.action]))

    if args.action == "random_gpu":
        benchmark_tensor_loading(
            load_random_tensor_on_gpu,
            warmup_cycle=False,
            action_repeat=action_repeat,
            device=device,
            output_base_folder=output_base_folder,
        )
        benchmark_tensor_loading(
            load_random_tensor_on_gpu,
            warmup_cycle=True,
            action_repeat=action_repeat,
            device=device,
            output_base_folder=output_base_folder,
        )
    elif args.action == "random_to_gpu":
        benchmark_tensor_loading(
            load_random_tensor_to_gpu,
            warmup_cycle=False,
            action_repeat=action_repeat,
            device=device,
            output_base_folder=output_base_folder,
        )
        benchmark_tensor_loading(
            load_random_tensor_to_gpu,
            warmup_cycle=True,
            action_repeat=action_repeat,
            device=device,
            output_base_folder=output_base_folder,
        )
    elif args.action == "single_image":
        benchmark_tensor_loading(
            load_local_image_to_gpu,
            warmup_cycle=False,
            action_repeat=action_repeat,
            device=device,
            output_base_folder=output_base_folder,
        )
        benchmark_tensor_loading(
            load_local_image_to_gpu,
            warmup_cycle=True,
            action_repeat=action_repeat,
            device=device,
            output_base_folder=output_base_folder,
        )
    elif args.action == "random_image":
        benchmark_tensor_loading(
            load_random_local_image_to_gpu,
            warmup_cycle=False,
            action_repeat=action_repeat,
            device=device,
            output_base_folder=output_base_folder,
        )
        benchmark_tensor_loading(
            load_random_local_image_to_gpu,
            warmup_cycle=True,
            action_repeat=action_repeat,
            device=device,
            output_base_folder=output_base_folder,
        )
    elif args.action == "mp":
        benchmark_tensor_loading(
            load_local_image_to_gpu,
            warmup_cycle=True,
            action_repeat=action_repeat,
            device=device,
            output_base_folder=output_base_folder,
        )
        mpap = MPActionPlayer(num_workers=8, pool_size=4)
        benchmark_tensor_loading(
            load_local_image_to_gpu,
            warmup_cycle=True,
            action_repeat=action_repeat,
            action_player=mpap,
            device=device,
            output_base_folder=output_base_folder,
        )
        benchmark_tensor_loading(
            load_random_tensor_to_gpu,
            warmup_cycle=True,
            action_repeat=action_repeat,
            action_player=mpap,
            device=device,
            output_base_folder=output_base_folder,
        )
        benchmark_tensor_loading(
            load_random_tensor_on_gpu,
            warmup_cycle=True,
            action_repeat=action_repeat,
            action_player=mpap,
            device=device,
            output_base_folder=output_base_folder,
        )
    else:
        parser.print_help()
        exit(2)


if __name__ == "__main__":
    main(*sys.argv[1:])
