.PHONY: help setup install lint format type-check test coverage clean run demo docker-build docker-run

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $1, $2}'

setup: ## Setup development environment
	@echo "Setting up development environment..."
	poetry install --with dev
	poetry run pre-commit install
	mkdir -p sandbox
	mkdir -p logs
	chmod +x scripts/setup.sh
	./scripts/setup.sh

install: ## Install dependencies only
	poetry install

lint: ## Run all linting
	poetry run ruff check app tests cli
	poetry run black --check app tests cli
	poetry run isort --check-only app tests cli

format: ## Format code
	poetry run black app tests cli
	poetry run isort app tests cli
	poetry run ruff check --fix app tests cli

type-check: ## Run type checking
	poetry run mypy app

test: ## Run tests
	poetry run pytest -v --cov=app --cov-report=html --cov-report=term-missing

coverage: test ## Generate coverage report
	@echo "Coverage report generated in htmlcov/index.html"

clean: ## Clean up generated files
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run: ## Run development server
	poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

demo: ## Run demo script
	poetry run python scripts/demo.py

docker-build: ## Build Docker image
	docker build -t autonomous-research-agent .

docker-run: ## Run with Docker Compose
	docker-compose up -d

docker-down: ## Stop Docker services
	docker-compose down

ci: lint type-check test ## Run CI pipeline locally

