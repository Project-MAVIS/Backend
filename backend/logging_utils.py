import logging
import os
from datetime import datetime
import pytz


class VerbosityLogger:
    """
    A logger class that supports verbosity levels and provides info and error logging capabilities.
    Logs with verbosity level less than or equal to the set level will be logged.
    """

    def __init__(self, verbosity: int = 1):
        self._verbosity = max(1, min(8, verbosity))  # Clamp between 1 and 8
        self._current_verbosity = 1

        # Configure root logger for Django logs
        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            root_logger.handlers.clear()
        root_logger.setLevel(logging.INFO)

        # Configure our app logger
        self._logger = logging.getLogger("django_app")
        if self._logger.hasHandlers():
            self._logger.handlers.clear()
        self._logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        # Create formatters with IST timezone
        ist_tz = pytz.timezone("Asia/Kolkata")

        class ISTFormatter(logging.Formatter):
            def formatTime(self, record, datefmt=None):
                dt = datetime.fromtimestamp(record.created, tz=ist_tz)
                return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S %Z")

        # Custom formatter for Django logs
        class DjangoFormatter(ISTFormatter):
            def format(self, record):
                if "django.server" in record.name or "django.request" in record.name:
                    return record.getMessage()
                return super().format(record)

        file_formatter = ISTFormatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z"
        )
        console_formatter = ISTFormatter("%(asctime)s %(message)s")
        django_formatter = DjangoFormatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z"
        )  # Empty format string as Django provides its own format

        # File handler - Change to use server.log
        file_handler = logging.FileHandler(os.path.join(log_dir, "server.log"))
        file_handler.setFormatter(file_formatter)
        self._logger.addHandler(file_handler)

        # Django file handler
        django_file_handler = logging.FileHandler(os.path.join(log_dir, "server.log"))
        django_file_handler.setFormatter(django_formatter)
        root_logger.addHandler(django_file_handler)

        # Console handlers
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        self._logger.addHandler(console_handler)

        django_console_handler = logging.StreamHandler()
        django_console_handler.setFormatter(django_formatter)
        root_logger.addHandler(django_console_handler)

        # Prevent propagation to root logger for our app logger
        self._logger.propagate = False

    def V(self, level: int) -> "VerbosityLogger":
        """Set the verbosity level for the next log message."""
        self._current_verbosity = max(1, min(8, level))
        return self

    def _should_log(self) -> bool:
        """Check if the current verbosity level should be logged."""
        return self._current_verbosity <= self._verbosity

    def _reset_verbosity(self) -> None:
        """Reset verbosity to default level after logging."""
        self._current_verbosity = 1

    def info(self, message: str) -> None:
        """
        Log an info message if the current verbosity level is less than or equal to
        the logger's verbosity setting.
        """
        if self._should_log():
            self._logger.info(f"[V{self._current_verbosity}] {message}")
        self._reset_verbosity()

    def error(self, message: str) -> None:
        """
        Log an error message regardless of verbosity level.
        """
        self._logger.error(f"[ERROR] {message}")
        self._reset_verbosity()


# Global logger instance
logger = VerbosityLogger()


def set_verbosity(level: int) -> None:
    """
    Set the global logger verbosity level.
    Args:
        level (int): Verbosity level (1-8)
    """
    global logger
    logger = VerbosityLogger(level)
