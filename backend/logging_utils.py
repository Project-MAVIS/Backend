import logging
from typing import cast

# Define custom logging levels
VERBOSE1 = 25
VERBOSE2 = 15
VERBOSE3 = 8
VERBOSE4 = 5
VERBOSE5 = 1

# Add custom level names
logging.addLevelName(VERBOSE1, "VERBOSE1")  # For basic messages
logging.addLevelName(VERBOSE2, "VERBOSE2")  # For more detailed messages
logging.addLevelName(VERBOSE3, "VERBOSE3")  # For debug messages
logging.addLevelName(VERBOSE4, "VERBOSE4")  # For trace messages
logging.addLevelName(VERBOSE5, "VERBOSE5")  # For most detailed messages


class VerbosityLogger(logging.Logger):
    def V(self, level: int) -> "VerbosityAdapter":
        """Returns a logger that only logs if verbosity level is high enough

        Args:
            level: Verbosity level (1-5)

        Returns:
            VerbosityAdapter: A logger adapter for the specified verbosity level
        """
        return VerbosityAdapter(self, level)


class VerbosityAdapter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def info(self, msg, *args, **kwargs):
        """Log with custom verbosity level"""
        # Calculate the actual logging level based on verbosity
        if self.level == 1:
            level = VERBOSE1
        elif self.level == 2:
            level = VERBOSE2
        elif self.level == 3:
            level = VERBOSE3
        elif self.level == 4:
            level = VERBOSE4
        elif self.level == 5:
            level = VERBOSE5
        else:
            level = logging.INFO

        self.logger.log(level, msg, *args, **kwargs)


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

        # Map verbose levels to their corresponding verbosity setting
        verbose_level_map = {
            VERBOSE1: 1,
            VERBOSE2: 2,
            VERBOSE3: 3,
            VERBOSE4: 4,
            VERBOSE5: 5,
        }

        # Get the required verbosity level for this record
        required_verbosity = verbose_level_map.get(record.levelno, 1)

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
# logger.V(1).info("Basic verbose message")  # Shows with -v 1 or higher
# logger.V(2).info("More detailed message")  # Shows with -v 2 or higher
# logger.V(3).info("Debug level message")  # Shows with -v 3 or higher
# logger.V(4).info("Trace level message")  # Shows with -v 4 or higher
# logger.V(5).info("Most detailed message")  # Shows with -v 5
