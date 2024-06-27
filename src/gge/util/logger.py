import logging


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and custom formatting."""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the root logger level

    if not logger.handlers:
        # Add the custom stream handler only if no handlers are configured yet
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  # Adjust as needed
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)

    # Add a NullHandler to prevent "No handlers could be found" errors in case of handler removals elsewhere
    logger.addHandler(logging.NullHandler())

    logger.propagate = False

    return logger