.PHONY: setup install clean format lint test coverage serve help

# Variables
PYTHON = python3
POETRY = poetry
VENV_NAME = .venv

help:  ## Show this help menu
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

keys:
	chmod +x keys.sh
	sh keys.sh

venv:  ## Create virtual environment in current directory
	test -d $(VENV_NAME) || $(PYTHON) -m venv $(VENV_NAME)
	$(POETRY) config virtualenvs.in-project true
	$(POETRY) env use $(VENV_NAME)/bin/python

setup: venv  ## Install project dependencies
	poetry config virtualenvs.in-project true
	$(POETRY) install --no-root

test:  ## Run tests
	$(PYTHON) manage.py test

start:  ## Start the development server
	$(POETRY) run python manage.py runserver

clean:  ## Remove Python file artifacts and cache directories
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".tox" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +

format:  ## Format code using black and isort
	$(POETRY) run black .
	$(POETRY) run isort .

lint:  ## Run linting checks
	$(POETRY) run flake8 .
	$(POETRY) run black . --check
	$(POETRY) run isort . --check
	$(POETRY) run mypy .

coverage:  ## Run tests with coverage report
	$(POETRY) run pytest --cov=./ --cov-report=term-missing

serve:  ## Run development server (if applicable)
	$(POETRY) run python manage.py runservers
