import logging
from rich.logging import RichHandler


def setup_logger(name: str = "morpheo", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    handler = RichHandler(rich_tracebacks=True, markup=True)
    formatter = logging.Formatter("%(name)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
