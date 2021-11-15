import json
import platform
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Optional

from dataset.s3_dataset import S3Dataset
from dataset.scratch_dataset import ScratchDataset
from dataset.t4c_s3_dataset import HDF5S3MODE
from dataset.t4c_s3_dataset import T4CDataset
from misc.logging_configuration import initialize_logging


def init_benchmarking(args: Namespace, action: str, loglevel: str = "INFO"):
    ts = datetime.now().strftime("%Y%m%df%H%M%S")
    output_base_folder = args.output_base_folder / f"{ts}_{action}"
    output_base_folder.mkdir(exist_ok=False, parents=True)
    initialize_logging(output_base_folder=output_base_folder, loglevel=loglevel)
    dump_metadata(args, output_base_folder)
    return output_base_folder


def dump_metadata(args, output_base_folder):
    with (output_base_folder / "metadata.json").open("w") as f:
        metadata = vars(args).copy()
        metadata["output_base_folder"] = metadata["output_base_folder"].name
        metadata.update(platform.uname()._asdict())
        json.dump(metadata, f)


def parse_args_file(json_file, dataset_type):
    json_args = json.load(open(json_file))
    for arg in list(json_args):
        if dataset_type not in arg and "file_download_url" in arg:
            del json_args[arg]
        elif "file_download_url" in arg:
            # rename the key
            json_args[arg.replace(dataset_type + "_", "")] = json_args.pop(arg)
    return json_args


def get_dataset(
    dataset: str,
    dataset_type: str = "val",
    use_cache=False,
    additional_args: Optional[Any] = None,
    limit: Optional[int] = None,
):
    if dataset == "t4c":
        # TODO magic constants... extract to cli... how to do in a generic way...
        dataset = T4CDataset(
            **json.load(open("s3_iarai_playground_t4c21.json")),
            index_file=Path(f"index-t4c-{dataset_type}.json"),
            limit=limit,
            mode=HDF5S3MODE[additional_args],
        )
    elif dataset == "s3":
        if use_cache == 1:
            use_cache = True
            endpoint = "http://localhost:6081"
        else:
            use_cache = False
            endpoint = "http://s3.amazonaws.com"
        dataset = S3Dataset(
            # TODO magic constants... extract to cli... how to do in a generic way...
            **parse_args_file("s3_iarai_playground_imagenet.json", dataset_type),
            index_file=Path(f"index-s3-{dataset_type}.json"),
            classes_file=Path(f"imagenet-{dataset_type}-classes.json"),
            limit=limit,
            endpoint_url=endpoint,
            use_cache=use_cache,
        )
    elif dataset == "scratch":
        dataset = ScratchDataset(
            index_file=Path(f"index-scratch-{dataset_type}.json"),
            classes_file=Path(f"imagenet-{dataset_type}-classes.json"),
            limit=limit,
        )
    print(f"Dataset loaded ... {dataset}, {dataset_type}, {len(dataset)}")
    return dataset
