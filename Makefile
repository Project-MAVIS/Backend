.PHONY: setup install clean format lint test coverage serve help venv

# Variables
PYTHON = python3
UV = uv
VENV_NAME = .venv

help: ## Show this help menu
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

keys: ## Set up keys for the project
	chmod +x keys.sh
	sh keys.sh

venv: ## Create virtual environment in current directory
	test -d $(VENV_NAME) || $(UV) venv

setup: venv ## Install project dependencies
	$(UV) run

test: ## Run tests
	$(PYTHON) manage.py test

start: ## Start the development server
	$(UV) run manage.py runserver

clean: ## Remove Python file artifacts and cache directories
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".tox" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +

format: ## Format code using black and isort
	$(UV) run black .
	$(UV) run isort .

lint: ## Run linting checks
	$(UV) run flake8 .
	$(UV) run black . --check
	$(UV) run isort . --check
	$(UV) run mypy .

coverage: ## Run tests with coverage report
	$(UV) run pytest --cov=./ --cov-report=term-missing

serve: ## Run development server (if applicable)
	$(PYTHON) manage.py runserver