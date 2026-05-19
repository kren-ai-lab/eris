import logging
from eris.logging import setup_logger
from eris.cli.main import app

if __name__ == "__main__":
    setup_logger(level=logging.INFO)
    app()
