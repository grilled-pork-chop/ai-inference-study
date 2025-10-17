# Makefile for Streaming LLM Chat project
.PHONY: help install install-dev clean format lint type-check test test-cov docker-build docker-run docker-stop docker-logs docker-clean

.DEFAULT_GOAL := help

UV := uv
DOCKER_COMPOSE := docker compose
DOCKER_IMAGE_NAME := streaming-llm-chat-api

help: ## Show this help message
	@echo "💬 Streaming LLM Chat API - Production Commands"
	@echo
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	$(UV) sync --frozen --no-dev

install-dev: ## Install development dependencies
	$(UV) sync --frozen --all-groups

run: ## Run the application locally (without Docker)
	$(UV) run start

clean: ## Clean cache files and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf dist/ build/ .ruff_cache/ .pytest_cache/ htmlcov/

format: ## Format code with ruff
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

lint: ## Run linting checks
	$(UV) run ruff check .
	$(UV) run ruff format --check .

type-check: ## Run type checking with mypy
	$(UV) run mypy src/

check: format lint type-check ## Run all quality checks

test: ## Run tests (placeholder, actual tests need to be written)
	@echo "No tests implemented yet."
	# $(UV) run pytest tests/ -v

test-cov: ## Run tests with coverage (placeholder)
	@echo "No tests implemented yet."
	# $(UV) run pytest tests/ -v --cov=src/streaming_llm_chat_api --cov-report=html --cov-report=term-missing

docker-build: ## Build Docker image for the API service
	docker build -t $(DOCKER_IMAGE_NAME):latest .

docker-run: ## Run full stack with Docker Compose
	$(DOCKER_COMPOSE) up -d

docker-stop: ## Stop Docker containers
	$(DOCKER_COMPOSE) down

docker-logs: ## Show application logs
	$(DOCKER_COMPOSE) logs -f api

docker-clean: ## Clean Docker resources
	$(DOCKER_COMPOSE) down -v --remove-orphans
	docker system prune -f

health: ## Check application health
	@echo "🏥 Checking application health..."
	curl -f http://localhost:8080/health || echo "❌ API not healthy"
