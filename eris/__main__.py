import logging

from eris.cli.main import app
from eris.logging import setup_logger

if __name__ == "__main__":
    setup_logger(level=logging.INFO)
    app()
