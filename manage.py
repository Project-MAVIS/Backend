#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from backend.logging_utils import set_verbosity


def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")

    # Add verbosity argument parsing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', '--logger',
        type=int,
        choices=range(1, 9),
        default=1,
        help='Set logger verbosity (1-8)'
    )

    # Parse known args to avoid conflicts with Django's argument parsing
    args, remaining_args = parser.parse_known_args()

    # Set logger verbosity
    set_verbosity(args.logger)

    # Continue with Django's normal startup
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line([sys.argv[0]] + remaining_args)


if __name__ == "__main__":
    main()
