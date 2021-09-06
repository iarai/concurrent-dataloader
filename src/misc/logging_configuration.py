import logging
import os
from logging import INFO


def log_format():
    return "[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s"


def initialize_logging(output_base_folder: str, loglevel=INFO):
    logging.basicConfig(level=loglevel, format=log_format())
    fh_root = logging.FileHandler(os.path.join(output_base_folder, f"out-{os.getpid()}.log"))
    logging.getLogger().addHandler(fh_root)

    fh_stopwatch = logging.FileHandler(os.path.join(output_base_folder, f"results-{os.getpid()}.log"))
    fh_stopwatch.setFormatter(logging.Formatter(""))
    logging.getLogger("stopwatch").addHandler(fh_stopwatch)
    # stopwatch always needs to log
    logging.getLogger("stopwatch").setLevel(logging.INFO)
