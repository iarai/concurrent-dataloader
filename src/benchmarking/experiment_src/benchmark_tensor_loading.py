import argparse
import logging
import sys
from functools import partial
from pathlib import Path
from typing import Callable
from typing import Type

import torch
from benchmarking.action_player.action_player import ActionPlayer
from benchmarking.action_player.mp_action_player import MPActionPlayer
from benchmarking.misc.init_benchmarking import init_benchmarking
from benchmarking.misc.random_generator import RandomGenerator
from PIL import Image
from torchvision import transforms
from benchmarking.misc.time_helper import stopwatch
import numpy as np

transform = transforms.Compose([transforms.CenterCrop(256), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),])
transform_resize = transforms.Compose([transforms.CenterCrop(256), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),])
rng = RandomGenerator()

IMAGE_PATH = "/iarai/home/ivan.svogor/git/storage-benchmarking/src/faster_dataloader/resources"

class MPLoading:
    def __init__(self, device="cuda:0", use_image=True, batch_size=4):
        self.device=device
        self.use_image=use_image
        self.batch_size=batch_size
        
    @stopwatch(trace_name="(1)-load_random_local_image_to_gpu", trace_level=1, strip_result=False)
    def load_random_local_image_to_gpu(self):
        # get all images and choose one at random
        image_path_list = list(Path(IMAGE_PATH).glob("*.JPEG"))
        num = rng.get_int(0, len(image_path_list) - 1)
        img_to_load = image_path_list[num]
        logging.debug(f"Opening local random image: {img_to_load}, rn: {num}")
        image = Image.open(img_to_load)
        if self.use_image:
            return image
        return self.just_tensor_load(image)

    @stopwatch(trace_name="(2)-just_tensor_load", trace_level=2, strip_result=False)
    def just_tensor_load(self, image):
        image_tensor = transform(image).cuda(self.device)
        return image_tensor

    @stopwatch(trace_name="(3)-open_random_batch", trace_level=3, strip_result=False)
    def open_random_batch(self):
        # img_tensors = []
        img_tensors = np.ndarray(shape=(self.batch_size, 256, 256), dtype=float, order='F')
        if self.use_image:
            for j, _ in enumerate(range(self.batch_size)):
                # no copying is performed here:
                # https://pytorch.org/docs/stable/generated/torch.as_tensor.html
                img_tensors[j] =transform_resize(self.load_random_local_image_to_gpu())
            batch = torch.as_tensor(img_tensors)
        else:
            batch = self.load_random_tensor_on_gpu()
        logging.debug(f"On device: {batch.is_cuda}, {batch.shape}")
        batch = self.just_tensor_load(batch)
        logging.debug(f"On device (after copy): {batch.is_cuda}")
        self.nothing(batch)
        
    @stopwatch(trace_name="(4)-just_batch_load", trace_level=4, strip_result=False)
    def just_tensor_load(self, batch):
        return batch.cuda(self.device)
        
    @stopwatch(trace_name="(4)-nothing", trace_level=4, strip_result=False)
    def nothing(self, batch):
        size = batch.shape

    @stopwatch(trace_name="(1)-load_local_image_to_gpu", trace_level=1, strip_result=False)
    def load_local_image_to_gpu(self) -> torch.Tensor:
        image_path_list = list(Path(IMAGE_PATH).glob("*.JPEG"))
        img_to_load = image_path_list[0]
        # logging.debug(f"Opening local image: {img_to_load}")
        image = Image.open(img_to_load)
        # perform transforms and send to GPU
        image_tensor = transform(image).cuda(self.device)
        return image_tensor

    @stopwatch(trace_name="(4)-create_tensor", trace_level=4, strip_result=False)
    def load_random_tensor_on_gpu(self) -> None:
        # creates a tensor directly on a GPU (same size as ImageNet)
        # check:
        #   https://towardsdatascience.com/7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259
        #   https://pytorch.org/docs/stable/notes/cuda.html
        return torch.rand(self.batch_size, 469, 387, device=torch.device(self.device))


    def load_random_tensor_to_gpu(self, device: str = "cuda:0") -> None:
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
    action_repeat: int = 10,
    action_player: Type[ActionPlayer] = None,
) -> None:
    if action_player is None:
        action_player = ActionPlayer()

    action_name = create_tensor_fn.__name__
    action_player.benchmark(
        action_name=action_name,
        action=partial(create_tensor_fn),
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
    parser.add_argument("--batch_size", type=int, default="64")
    parser.add_argument("--pool_size", type=int, default=4)

    return parser


# TODO we should not use data from warmup cycle!!!
def main(*args):
    parser = handle_arguments()
    args = parser.parse_args(args)
    device = args.device
    batch_size = args.batch_size
    pool_size = args.pool_size
    output_base_folder = init_benchmarking(args, action="_".join(["benchmark_tensor_loading", args.action]))

    if args.action == "random_batch":
        mp_loading = MPLoading(device=device, use_image=True, batch_size=batch_size)   
        mp_loading.open_random_batch()
    elif args.action == "random_batch_on_device":
        mp_loading = MPLoading(device=device, use_image=False, batch_size=batch_size)   
        mp_loading.open_random_batch()
    elif args.action == "random_batch_mp":
        mpap = MPActionPlayer(pool_size=pool_size)
        mp_loading = MPLoading(device=device, use_image=True, batch_size=batch_size)   
        benchmark_tensor_loading(
            create_tensor_fn = mp_loading.open_random_batch,
            action_player=mpap,
            output_base_folder=output_base_folder,
        )

    else:
        parser.print_help()
        exit(2)


if __name__ == "__main__":
    main(*sys.argv[1:])
