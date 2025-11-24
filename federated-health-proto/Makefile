.PHONY: setup start stop test package-client clean help

help:
	@echo "Federated Learning Application - Available Commands"
	@echo "=================================================="
	@echo "setup             - Install dependencies and setup environment"
	@echo "start             - Start all services with Docker Compose"
	@echo "start-cpu         - Start with CPU-limited resources"
	@echo "stop              - Stop all services"
	@echo "test              - Run unit and integration tests"
	@echo "test-unit         - Run only unit tests"
	@echo "test-integration  - Run only integration tests"
	@echo "package-client    - Package Python client as standalone executable"
	@echo "download-data     - Download Kaggle datasets"
	@echo "demo              - Run full demo with simulated clients"
	@echo "clean             - Clean up Docker volumes and build artifacts"
	@echo "logs              - Tail Docker Compose logs"

setup:
	@echo "Setting up environment..."
	cp .env.example .env
	@echo "Please configure your .env file with proper credentials"
	@echo "Installing server dependencies..."
	cd server && pip install -r requirements.txt
	@echo "Installing client dependencies..."
	cd client && pip install -r requirements.txt
	@echo "Installing frontend dependencies..."
	cd web-ui && npm install

start:
	@echo "Starting Federated Learning services..."
	docker-compose up --build -d
	@echo "Services started! Access:"
	@echo "  Frontend: http://localhost:3000"
	@echo "  API: http://localhost:8000"
	@echo "  MinIO Console: http://localhost:9001"

start-cpu:
	@echo "Starting with CPU resource limits..."
	docker-compose -f docker-compose.cpu.yml up --build -d

stop:
	docker-compose down

logs:
	docker-compose logs -f

test:
	@echo "Running all tests..."
	cd tests && pytest -v

test-unit:
	@echo "Running unit tests..."
	cd tests && pytest -v unit/

test-integration:
	@echo "Running integration tests..."
	cd tests && pytest -v integration/

package-client:
	@echo "Packaging client as executable..."
	cd client && bash package_client.sh

download-data:
	@echo "Downloading Kaggle datasets..."
	bash scripts/download_datasets.sh

demo:
	@echo "Running demo simulation..."
	bash scripts/run_demo.sh

clean:
	@echo "Cleaning up..."
	docker-compose down -v
	rm -rf models/ data/ minio-data/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
