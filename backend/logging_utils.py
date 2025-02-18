import logging
from typing import cast, Any


class VerbosityLogger(logging.Logger):
    def V(self, level: int) -> "VerbosityAdapter":
        """Returns a logger that only logs if verbosity level is high enough

        Args:
            level: Verbosity level (1-5)

        Levels:
            0: NOTSET
            1: DEBUG
            2: INFO
            3: WARNING
            4: ERROR
            5: CRITICAL

        Returns:
            VerbosityAdapter: A logger adapter for the specified verbosity level
        """
        return VerbosityAdapter(self, level)


class VerbosityAdapter:
    def __init__(self, logger: VerbosityLogger, level: int) -> None:
        self.logger = logger
        self.level = level

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log with custom verbose level

        Args:
            msg: The message to log
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        # Calculate the actual logging level based on verbosity
        if self.level == 0:
            level = logging.NOTSET
        elif self.level == 1:
            level = logging.DEBUG
        elif self.level == 2:
            level = logging.INFO
        elif self.level == 3:
            level = logging.WARNING
        elif self.level == 4:
            level = logging.ERROR
        elif self.level == 5:
            level = logging.CRITICAL
        else:
            level = logging.INFO

        self.logger.log(level, msg, *args, **kwargs)

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log an error message (always shown regardless of verbosity level)

        Args:
            msg: The message to log
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        self.logger.error(msg, *args, **kwargs)


class VerbosityFilter(logging.Filter):
    def __init__(self, verbosity_level):
        super().__init__()
        self.verbosity_level = verbosity_level

    def filter(self, record):
        # Allow all standard logging levels
        if record.levelno in (
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ):
            return True

        # Get the required verbosity level for this record
        required_verbosity = self.verbosity_level

        # Allow the record if the current verbosity level is high enough
        return self.verbosity_level >= required_verbosity


# Register the custom logger class
logging.setLoggerClass(VerbosityLogger)


def get_verbose_logger(name: str) -> VerbosityLogger:
    """Get a logger instance with verbosity support

    Args:
        name: The name of the logger

    Returns:
        VerbosityLogger: A logger with verbosity support
    """
    return cast(VerbosityLogger, logging.getLogger(name))


# Example usage

# logger = logging.getLogger("server_log")

# These will only show up if the verbosity level is high enough
# logger.V(0).info("Basic verbose message")  # Shows with -v 0 or higher
# logger.V(1).info("Basic verbose message")  # Shows with -v 1 or higher
# logger.V(2).info("More detailed message")  # Shows with -v 2 or higher
# logger.V(3).info("Debug level message")  # Shows with -v 3 or higher
