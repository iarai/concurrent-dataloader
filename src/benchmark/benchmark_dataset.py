import argparse
import json
import logging
import sys
from pathlib import Path

from action_player.action_player import ActionPlayer
from action_player.mp_action_player import MPActionPlayer
from dataset.indexed_dataset import IndexedDataset
from dataset.s3_dataset import S3Dataset
from dataset.scratch_dataset import ScratchDataset
from dataset.t4c_s3_dataset import HDF5S3MODE
from dataset.t4c_s3_dataset import T4CDataset
from main import init_benchmarking


# main function that defines the testing order ... e.g. index, load, save


def benchmark_dataset(
    dataset: IndexedDataset,
    output_base_folder: Path,
    skip_indexing: bool = True,
    mp: bool = False,
    pool_size: int = 5,
    num_load_index=5,
    num_get_random_item=10000,
) -> None:
    # TODO once we have all options from cli, we may get rid of this
    with (output_base_folder / "benchmark_dataset.json").open("w") as f:
        json.dump(
            {
                # TODO log params as well
                "dataset": str(dataset),
                "mp": mp,
                "pool_size": pool_size,
                "num_load_index": num_load_index,
                "num_get_random_item": num_get_random_item,
            },
            f,
        )
    if not mp:
        action_player = ActionPlayer()
    else:
        assert skip_indexing, "Indexing cannot be performed by Multi-Processing ActionPlayer"
        action_player = MPActionPlayer(pool_size=pool_size)

    # ls (index) all images
    if not skip_indexing:
        logging.info("Indexing")
        # TODO use param
        action_player.benchmark("indexing", dataset.index_all, 5)

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
    parser.add_argument(
        "-a",
        "--action",
        help="An option to benchmark (s3, scratch, random_gpu, random_to_gpu, random_image)",
        default="random_gpu",
    )
    parser.add_argument("--output_base_folder", type=Path, default=Path("benchmark_output"))
    parser.add_argument("-args", "--args", nargs="+", help="Additional arguments")
    return parser


def main(*args):
    parser = handle_arguments()
    args = parser.parse_args(args)
    output_base_folder = init_benchmarking(args, action="_".join(["benchmark_dataset", args.action]))

    # -------------------------------------
    # benchmark_dataset
    # -------------------------------------
    if args.action == "t4c":
        benchmark_dataset(
            # TODO magic constants... extract to cli... how to do in a generic way...
            dataset=T4CDataset(
                **json.load(open("s3_iarai_playground_t4c21.json")),
                index_file=Path("index-t4c.json"),
                mode=HDF5S3MODE[args.args[0]],
            ),
            skip_indexing=True,
            mp=True,
            # TODO magic constants... extract to cli... how to do in a generic way...
            num_load_index=0,
            num_get_random_item=5,
            pool_size=5,
            output_base_folder=output_base_folder,
        )

    elif args.action == "s3":
        benchmark_dataset(
            S3Dataset(
                # TODO magic constants... extract to cli... how to do in a generic way...
                **json.load(open("s3_iarai_playground_imagenet.json")),
                index_file=Path("index-s3-val.json"),
            ),
            skip_indexing=True,
            mp=False,
            output_base_folder=output_base_folder,
            num_load_index=0,
            num_get_random_item=55,
        )
    elif args.action == "scratch":
        benchmark_dataset(
            # TODO magic constants... extract to cli... how to do in a generic way...
            dataset=ScratchDataset(index_file=Path("index-scratch-val.json")),
            output_base_folder=output_base_folder,
            mp=True,
            pool_size=4,
            num_load_index=0,
            num_get_random_item=55,
        )
    else:
        parser.print_help()
        exit(2)


if __name__ == "__main__":
    main(*sys.argv[1:])
