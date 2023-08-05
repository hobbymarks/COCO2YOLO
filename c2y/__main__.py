"""
main entry
"""
import signal
import sys

import fire
from loguru import logger

from c2y import Xcoco


def signal_handler(sig, frame):
    """
    signal handler
    """
    if sig == signal.SIGINT:
        logger.info(f"Func:{frame.f_code.co_name},CodeLine:{frame.f_lineno}")
        sys.exit(0)
    else:
        logger.info(f"Signal:{sig}")


def main_entry():
    """
    main entry
    """
    logger.remove()
    # Log to file, rotate at 10 MB
    logger.add("logs/c2y_log.log", level="TRACE", rotation="1 MB")
    # Handle signal
    signal.signal(signal.SIGINT, signal_handler)

    fire.Fire(Xcoco)


if __name__ == "__main__":
    main_entry()
