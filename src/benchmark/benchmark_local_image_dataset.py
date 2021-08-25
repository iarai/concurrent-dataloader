import logging
from pathlib import Path
from typing import Callable
from typing import Type

import torch
from action_player.action_player import ActionPlayer
from misc.random_generator import RandomGenerator
from PIL import Image
from torchvision import transforms

transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),])
rng = RandomGenerator()

IMAGE_PATH = "resources/"


def load_random_local_image_to_gpu() -> torch.Tensor:
    # get all images and choose one at random
    image_path_list = list(Path(IMAGE_PATH).glob("*.JPEG"))
    num = rng.get_int(0, len(image_path_list) - 1)
    img_to_load = image_path_list[num]
    logging.debug(f"Oppening local random image: {img_to_load}, rn: {num}")
    image = Image.open(img_to_load)
    # perform transforms and send to GPU
    image_tensor = transforms(image).cuda()
    return image_tensor


def load_local_image_to_gpu() -> torch.Tensor:
    image_path_list = list(Path(IMAGE_PATH).glob("*.JPEG"))
    img_to_load = image_path_list[0]
    logging.debug(f"Oppening local image: {img_to_load}")
    image = Image.open(img_to_load)
    # perform transforms and send to GPU
    image_tensor = transforms(image).cuda()
    return image_tensor


def load_random_tensor_on_gpu() -> None:
    # creates a tensor directly on a GPU (same size as ImageNet)
    torch.rand(469, 387, device=torch.device("cuda:0"))


def load_random_tensor_to_gpu() -> None:
    # creates a tensor directly on a GPU (same size as ImageNet)
    # check:
    #   https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
    #   https://pytorch.org/docs/stable/notes/cuda.html
    torch.rand(469, 387).cuda()


def benchmark_tensor_loading(
    create_tensor_fn: Callable,
    warmup_cycle: bool = False,
    action_repeat: int = 10,
    action_player: Type[ActionPlayer] = None,
    verbose=False,
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
            torch.rand(256, 256, device=torch.device("cuda:0"))
        action_name = action_name + "_with_warmup"
    action_player.benchmark(action_name=action_name, action=create_tensor_fn, repeat=action_repeat, verbose=verbose)
