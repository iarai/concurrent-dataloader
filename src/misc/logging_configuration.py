import logging
from logging import INFO


def log_format():
    return "[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s"


def initialize_logging(loglevel=INFO):
    logging.basicConfig(level=loglevel, format=log_format())
