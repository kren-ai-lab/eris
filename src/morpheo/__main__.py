import logging
from morpheo.logging import setup_logger
from morpheo.cli.main import app

if __name__ == "__main__":
    setup_logger(level=logging.INFO)
    app()
