import json
import logging
import platform
from argparse import Namespace
from datetime import datetime

from misc.logging_configuration import initialize_logging


def init_benchmarking(args: Namespace, action: str):
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    output_base_folder = args.output_base_folder / f"{ts}_{action}"
    output_base_folder.mkdir(exist_ok=False, parents=True)
    initialize_logging(output_base_folder=output_base_folder, loglevel=logging.INFO)
    dump_metadata(args, output_base_folder)
    return output_base_folder


def dump_metadata(args, output_base_folder):
    with (output_base_folder / "metadata.json").open("w") as f:
        metadata = vars(args).copy()
        metadata["output_base_folder"] = metadata["output_base_folder"].name
        metadata.update(platform.uname()._asdict())
        json.dump(metadata, f)
