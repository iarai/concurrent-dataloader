import argparse
import logging
import sys
import os
from pathlib import Path

from benchmarking.action_player.action_player import ActionPlayer
from benchmarking.action_player.mp_action_player import MPActionPlayer
from faster_dataloader.dataset.indexed_dataset import IndexedDataset
from benchmarking.misc.init_benchmarking import init_benchmarking
from benchmarking.misc.init_benchmarking import get_dataset


def benchmark_dataset(
        dataset: IndexedDataset,
        output_base_folder: Path,
        pool_size: int = 5,
        num_index_all=0,
        num_load_index=5,
        num_get_random_item=10000,
) -> None:
    if pool_size == 0:
        action_player = ActionPlayer()
    else:
        assert num_index_all == 0, "Indexing cannot be performed by Multi-Processing ActionPlayer"
        action_player = MPActionPlayer(pool_size=pool_size)

    # ls (index) all images
    logging.info("Indexing")
    action_player.benchmark(
        "indexing", action=dataset.index_all, repeat=num_index_all, output_base_folder=output_base_folder
    )

    # load random images
    action_player.benchmark(
        action_name="loading_random",
        action=dataset.get_random_item,
        repeat=num_get_random_item,
        output_base_folder=output_base_folder,
    )

    # load index from file
    logging.info("Loading index... ")
    dataset.load_index()

    action_player.benchmark(
        action_name="load_index",
        action=dataset.load_index,
        repeat=num_load_index,
        output_base_folder=output_base_folder,
    )
    logging.info(f"Loading index... Done. Len {dataset.__len__()}")


def handle_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base_folder", type=Path, default=Path("benchmark_output"))
    parser.add_argument(
        "--dataset", help="An option to benchmark (s3, scratch,)", default="s3",
    )
    parser.add_argument(
        "--num_get_random_item",
        type=int,
        help="An option to benchmark (s3, scratch, random_gpu, random_to_gpu, random_image)",
        default=55,
    )
    parser.add_argument(
        "--num_load_index",
        type=int,
        help="An option to benchmark (s3, scratch, random_gpu, random_to_gpu, random_image)",
        default=0,
    )
    parser.add_argument(
        "--num_index_all",
        type=int,
        help="An option to benchmark (s3, scratch, random_gpu, random_to_gpu, random_image)",
        default=0,
    )
    parser.add_argument(
        "--pool_size",
        type=int,
        help="An option to benchmark (s3, scratch, random_gpu, random_to_gpu, random_image)",
        default=5,
    )
    parser.add_argument("--args", nargs="+", help="Additional arguments")
    parser.add_argument("--loglevel", default="INFO", help="Additional arguments")
    return parser


def main(*args):
    parser = handle_arguments()
    args = parser.parse_args(args)
    dataset = args.dataset
    output_base_folder = init_benchmarking(
        args, action="_".join(["benchmark_dataset", dataset]), loglevel=args.loglevel
    )
    logging.info("==================== benchmark_dataset %s ==========================================", vars(args))
    benchmark_args = vars(args).copy()
    benchmark_args.pop("dataset")
    benchmark_args.pop("args")
    benchmark_args.pop("loglevel")
    benchmark_args["output_base_folder"] = output_base_folder

    # -------------------------------------
    # benchmark_dataset
    # -------------------------------------
    base_folder = os.path.dirname(__file__)
    s3_credential_file = os.path.join(base_folder, "../credentials_and_indexes/s3_iarai_playground_imagenet.json")
    train_dataset_index = f"../credentials_and_indexes/index-{args.dataset}-train.json"

    # additional_args = args.args
    dataset = get_dataset(dataset=dataset,
                          dataset_type="train",
                          use_cache=False,
                          index_file=Path(os.path.join(base_folder,
                                                       train_dataset_index)),
                          classes_file=Path(os.path.join(base_folder,
                                                         "../credentials_and_indexes/imagenet-train-classes.json")),
                          s3_credential_file=s3_credential_file)

    if dataset is None:
        parser.print_help()
        exit(2)

    benchmark_dataset(dataset=dataset, **benchmark_args,)


if __name__ == "__main__":
    main(*sys.argv[1:])
