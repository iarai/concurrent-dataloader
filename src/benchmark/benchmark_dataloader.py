import argparse
import json
import logging
import sys
from functools import partial
from pathlib import Path
from typing import List

import torch
from action_player.action_player import ActionPlayer
from data_loader.async_data_loader import AsynchronousLoader
from dataset.s3_dataset import S3Dataset
from main import init_benchmarking
from misc.logging_configuration import initialize_logging
from misc.time_helper import stopwatch
from torch.functional import Tensor
from torch_overrides.dataloader import DataLoader
from torch_overrides.worker import _worker_loop


@stopwatch("(2)-load_single")
def load_single(dataloader: DataLoader) -> None:
    try:
        _ = next(iter(dataloader))
    except (StopIteration, EOFError) as e:
        logging.info(f"Exception raised: {str(e)}")


@stopwatch("(2)-load_all")
def load_all(dataloader: DataLoader) -> None:
    try:
        # loading the data, replace with i, batch -> logging.info(f"{len(batch)}, {i}, {len(dataloader)}")
        for _, _ in enumerate(dataloader):
            pass
    except (StopIteration, EOFError) as e:
        logging.info(f"Exception raised : {e}")
    except Exception as e:
        logging.info(f"Exception raised : {e}")


def collate(batch: List) -> Tensor:
    imgs = [item for item in batch]  # noqa
    return imgs


@stopwatch("(1)-benchmark")
def benchmark_dataloader(
    batch_size: int,
    num_workers: int,
    data_loader_type: str,
    output_base_folder: Path,
    repeat: int = 10,
    prefetch_factor=2,
    num_fetch_workers=4,
    limit=50,
    device: str = "cuda",
) -> None:
    action_player = ActionPlayer()

    # TODO once we have all options from cli, we may get rid of this
    with (output_base_folder / "benchmark_dataloader.json").open("w") as f:
        json.dump(
            {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "data_loader_type": data_loader_type,
                "repeat": repeat,
                "prefetch_factor": prefetch_factor,
                "num_fetch_workers": num_fetch_workers,
                "limit": limit,
                "device": device,
            },
            f,
        )

    _dataset = S3Dataset(
        # TODO magic constants
        bucket_name="iarai-playground",
        index_file=Path("index-s3-val.json"),
        index_file_download_url="s3://iarai-playground/scratch/imagenet/index-s3-val.json",
        limit=limit,
    )
    _dataset.load_index()

    if data_loader_type == "async":
        data_loader = AsynchronousLoader(
            data=_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            device=device,
            collate_fn=collate,
            prefetch_factor=prefetch_factor,
            num_fetch_workers=num_fetch_workers,
        )
    else:
        data_loader = DataLoader(
            dataset=_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate,
            prefetch_factor=prefetch_factor,
            num_fetch_workers=num_fetch_workers,
        )

    # TODO should we do this in a central place?
    # override the _worker_loop to inject @stopwatch
    assert len(logging.getLogger("stopwatch").handlers) > 0
    torch.utils.data._utils.worker._worker_loop = partial(
        _worker_loop,
        initializer=partial(
            initialize_logging, loglevel=logging.getLogger().getEffectiveLevel(), output_base_folder=output_base_folder,
        ),
    )

    logging.info(f"Warmup ... batch {batch_size}, workers {num_workers}")
    action_player.benchmark(
        "loading_with_dataloader_warmup", lambda: load_single(data_loader), repeat=repeat,
    )
    logging.info("Warmup -- end")

    # real benchmark
    action_player.benchmark("loading_with_dataloader", lambda: load_all(data_loader), repeat=repeat)


def handle_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base_folder", type=Path, default=Path("benchmark_output"))
    parser.add_argument("--batch_size", type=int, default=50, help="Additional arguments")
    parser.add_argument("--num_workers", type=int, default=2, help="Additional arguments")
    parser.add_argument("--data_loader_type", type=str, default="sync", help="Additional arguments")
    parser.add_argument("--num_fetch_workers", type=int, default=1, help="Additional arguments")
    parser.add_argument("--repeat", type=int, default=50, help="Additional arguments")
    parser.add_argument("--limit", type=int, default=50, help="Additional arguments")
    return parser


def main(*args):
    parser = handle_arguments()
    args = parser.parse_args(args)

    output_base_folder = init_benchmarking(
        args=args, action="_".join([str(args.batch_size), str(args.num_workers), args.data_loader_type])
    )
    args = vars(args)
    args["output_base_folder"] = output_base_folder

    benchmark_dataloader(**args)


if __name__ == "__main__":
    main(*sys.argv[1:])
