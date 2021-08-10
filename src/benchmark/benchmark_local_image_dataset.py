import logging
from typing import Callable
from typing import Type

import torch
from action_player.action_player import ActionPlayer
from PIL import Image
from torchvision import transforms

transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),])


def load_random_image_to_gpu() -> torch.Tensor:
    image = Image.open("resources/collie.jpeg")
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
    action_player.benchmark(action_name, create_tensor_fn, action_repeat)
