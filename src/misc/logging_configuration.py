import logging
import os
from logging import INFO


def log_format():
    return "[%(asctime)s][%(levelname)s][%(process)df][%(pathname)s:%(funcName)s:%(lineno)df] %(message)s"


def initialize_logging(output_base_folder: str, loglevel=INFO):
    logging.basicConfig(level=loglevel, format=log_format())

    root = logging.getLogger()
    root.setLevel(loglevel)
    fh_root = logging.FileHandler(os.path.join(output_base_folder, f"out-{os.getpid()}.log"))
    root.addHandler(fh_root)

    fh_stopwatch = logging.FileHandler(os.path.join(output_base_folder, f"results-{os.getpid()}.log"))
    fh_stopwatch.setFormatter(logging.Formatter(""))
    logging.getLogger("stopwatch").addHandler(fh_stopwatch)
    # stopwatch always needs to log but do not propagate to root, as messages are passed directly
    # to the ancestor loggers’ handlers - neither the level nor filters of the ancestor loggers in question
    # are considered.
    # https://stackoverflow.com/questions/60196327/python-logging-debug-messages-logged-to-stderr-even-though-handler-level-is-inf
    logging.getLogger("stopwatch").setLevel(logging.DEBUG)
    logging.getLogger("stopwatch").propagate = False

    timeline_log = logging.FileHandler(os.path.join(output_base_folder, f"timeline-{os.getpid()}.log"))
    timeline_log.setFormatter(logging.Formatter(""))
    logging.getLogger("timeline").addHandler(timeline_log)
    logging.getLogger("timeline").setLevel(logging.DEBUG)
    logging.getLogger("timeline").propagate = False
