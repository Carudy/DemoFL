import logging
from functools import partial
from logging import INFO

__all__ = ['logger', 'log']

LOGGER_NAME = "DemoFL"
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def get_logger(name=LOGGER_NAME, level=LOG_LEVEL, log_format=LOG_FORMAT):
    _logger = logging.getLogger(name)
    if not _logger.handlers:
        _logger.setLevel(level=level)
        formatter = logging.Formatter(log_format)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        _logger.addHandler(sh)
    return _logger


logger = get_logger(LOGGER_NAME)
logger.propagate = False
log = partial(logger.log, INFO)
