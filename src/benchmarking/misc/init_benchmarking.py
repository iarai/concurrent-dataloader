import json
import platform
from argparse import Namespace
from datetime import datetime
from typing import Any
from typing import Optional

from benchmarking.misc.logging_configuration import initialize_logging
from concurrent_dataloader.dataset.s3_dataset import S3Dataset
from concurrent_dataloader.dataset.scratch_dataset import ScratchDataset


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
    index_file: str,
    classes_file: str,
    use_cache: False,
    dataset_type: str = "val",
    additional_args: Optional[Any] = None,
    limit: Optional[int] = None,
    s3_credential_file: Optional[str] = None,
    flavor: Optional[str] = None,
):
    if dataset == "s3" or dataset == "ceph-os":
        if use_cache == 1:
            use_cache = True
            endpoint = "http://localhost:6081" # for CEPH Object Store this needs to be changed to different address 
        else:
            use_cache = False
            if dataset == "ceph-os":
                endpoint = "http://10.0.2.1:80"
            else:
                endpoint = "http://s3.amazonaws.com"
        dataset = S3Dataset(
            # TODO magic constants... extract to cli... how to do in a generic way...
            **parse_args_file(s3_credential_file, dataset_type),
            index_file=index_file,
            classes_file=classes_file,
            limit=limit,
            endpoint_url=endpoint,
            use_cache=use_cache,
            flavor=flavor
        )
    elif dataset == "scratch":
        dataset = ScratchDataset(index_file=index_file, classes_file=classes_file, limit=limit,)
    elif dataset == "glusterfs": # also works for ceph 
        dataset = ScratchDataset(
            index_file=index_file, classes_file=classes_file, limit=limit, local_path="/iarai/home/ivan.svogor/temp"
        )
    print(f"Dataset loaded ... {dataset}, {dataset_type}, {len(dataset)}")
    return dataset
