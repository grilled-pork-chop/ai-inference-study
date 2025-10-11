import logging

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


def configure_logger(level: str) -> None:
    """Configure the logger with the specified log level."""

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(fmt=DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[stream_handler],
    )
