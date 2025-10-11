# Makefile for Image Classification API project
.PHONY: help install install-dev clean format lint type-check test test-cov docker-build docker-run benchmark

.DEFAULT_GOAL := help

UV := uv
DOCKER_COMPOSE := docker-compose
DOCKER_IMAGE_NAME := image-classification-api

help: ## Show this help message
	@echo "🚀 Image Classification API - Production Commands"
	@echo
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	$(UV) sync --frozen --no-dev

install-dev: ## Install development dependencies
	$(UV) sync --frozen

run: ## Run the application
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

security-check: ## Run security checks
	$(UV) run bandit -r src/
	$(UV) run safety check

check: format lint type-check security-check ## Run all quality checks

test: ## Run tests
	$(UV) run pytest tests/ -v

test-cov: ## Run tests with coverage
	$(UV) run pytest tests/ -v --cov=src/image_classification_api --cov-report=html --cov-report=term-missing

benchmark: ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	$(UV) run python scripts/benchmark.py

convert-onnx: ## Convert ResNet50 model to ONNX
	$(UV) run python scripts/convert_to_onnx.py

setup-model: ## Setup Triton model repository
	mkdir -p model_repository/resnet50_onnx/1
	@echo "✅ Model repository structure created"
	@echo "   Place model.onnx in model_repository/resnet50_onnx/1/"

docker-build: ## Build Docker image
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
	curl -f http://localhost:8080/api/v1/health || echo "❌ API not healthy"

metrics: ## Show Prometheus metrics
	@echo "📊 Fetching Prometheus metrics..."
	curl -s http://localhost:8080/metrics | head -20


# Quick Start
quick-start: install-dev setup-model docker-run ## Quick start for development
	@echo
	@echo "🚀 Development environment ready!"
	@echo "   📝 API Documentation: http://localhost:8000/docs"
	@echo "   💖 API Health: http://localhost:8000/api/v1/health"
	@echo "   📊 Metrics: http://localhost:8000/metrics"
	@echo "   📈 Grafana: http://localhost:3000 (admin/admin123)"
	@echo "   🔍 Prometheus: http://localhost:9090"
	@echo
	@echo "🧪 Test the API:"
	@echo '   curl -X POST "http://localhost:8080/api/v1/classify" -H "Content-Type: multipart/form-data" -F "file=@your_image.jpg"'
	@echo
	@echo "📖 Available commands: make help"
