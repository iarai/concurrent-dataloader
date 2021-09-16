import argparse
import logging
import sys
from functools import partial
from pathlib import Path
from typing import List
from typing import Optional

import torch
from action_player.action_player import ActionPlayer
from data_loader.async_data_loader import AsynchronousLoader
from dataset.indexed_dataset import IndexedDataset
from main import get_dataset
from main import init_benchmarking
from misc.logging_configuration import initialize_logging
from misc.time_helper import stopwatch
from torch.functional import Tensor
from torch_overrides.dataloader import DataLoader
from torch_overrides.worker import _worker_loop


@stopwatch(trace_name="(2)-load_all", trace_level=2)
def load_all(dataloader: DataLoader, num_batches: Optional[int] = None) -> None:
    try:
        for i, _ in enumerate(dataloader):
            if num_batches is not None and i == num_batches - 1:
                break
            pass
    except (StopIteration, EOFError) as e:
        logging.info(f"Exception raised : {e}")
    except Exception as e:
        logging.info(f"Exception raised : {e}")


def collate(batch: List) -> Tensor:
    imgs = [item for item in batch]  # noqa
    return imgs


@stopwatch(trace_name="(1)-benchmark", trace_level=1)
def benchmark_dataloader(
    dataset: IndexedDataset,
    batch_size: int,
    num_workers: int,
    data_loader_type: str,
    output_base_folder: Path,
    repeat: int = 10,
    prefetch_factor=2,
    num_fetch_workers=4,
    device: str = "cuda",
    shuffle: bool = False,
    num_batches: Optional[int] = None,
    fetch_impl: Optional[str] = None,
    batch_pool: Optional[int] = None,
) -> None:
    action_player = ActionPlayer()

    dataset.load_index()

    if data_loader_type == "async":
        data_loader = AsynchronousLoader(
            data=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            device=device,
            collate_fn=collate,
            prefetch_factor=prefetch_factor,
            num_fetch_workers=num_fetch_workers,
            fetch_impl=fetch_impl,
            batch_pool=batch_pool,
        )
    else:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=collate,
            prefetch_factor=prefetch_factor,
            num_fetch_workers=num_fetch_workers,
            fetch_impl=fetch_impl,
            batch_pool=batch_pool,
        )

    # TODO should we do this in a central place?
    # override the _worker_loop to inject @stopwatch
    torch.utils.data._utils.worker._worker_loop = partial(
        _worker_loop,
        initializer=partial(
            initialize_logging, loglevel=logging.getLogger().getEffectiveLevel(), output_base_folder=output_base_folder,
        ),
    )
    # real benchmark
    action_player.benchmark(
        "loading_with_dataloader", lambda: load_all(data_loader, num_batches=num_batches), repeat=repeat
    )


def handle_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base_folder", type=Path, default=Path("benchmark_output"))
    parser.add_argument("--dataset", type=str, default="s3")
    parser.add_argument("--batch_size", type=int, default=50, help="Additional arguments")
    parser.add_argument("--num_workers", type=int, default=1, help="Additional arguments")
    parser.add_argument(
        "--data_loader_type", type=str, default="sync", help="sync/async, " "async is CUDA stream processing"
    )
    parser.add_argument("--num_fetch_workers", type=int, default=16, help="Additional arguments")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Additional arguments")
    parser.add_argument("--repeat", type=int, default=1, help="Additional arguments")
    parser.add_argument("--num_batches", type=int, default=5, help="None means full dataset")
    parser.add_argument("--shuffle", type=bool, default=True, help="Additional arguments")
    parser.add_argument("--fetch_impl", type=str, default="threaded", help="threaded or acyncio")
    parser.add_argument("--batch_pool", type=int, default=10, help="Batch pool to collect together")
    return parser


def main(*args):
    parser = handle_arguments()
    args = parser.parse_args(args)

    output_base_folder = init_benchmarking(
        args=args,
        action="_".join(
            [
                "benchmark_dataloader",
                str(args.dataset),
                str(args.batch_size),
                str(args.num_workers),
                str(args.num_fetch_workers),
                args.data_loader_type,
            ]
        ),
    )
    args = vars(args)
    args["output_base_folder"] = output_base_folder
    args["dataset"] = get_dataset(dataset=args["dataset"], additional_args=[])
    benchmark_dataloader(**args)


if __name__ == "__main__":
    main(*sys.argv[1:])


# TODO with num_workers > 0, why do the logs of the workers go into the main processes log?
#  Is this PyTorch doing something?
