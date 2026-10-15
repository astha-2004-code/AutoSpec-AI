# AutoSpec AI — Development & Operations Makefile

.PHONY: help install lint test evaluate run-api run-ui docker-build docker-up docker-down clean

help:
	@echo "AutoSpec AI — Command Reference"
	@echo "--------------------------------------------------------"
	@echo "  make install       Install package in editable mode with dev dependencies"
	@echo "  make lint          Run Ruff static analysis and linting checks"
	@echo "  make test          Execute automated pytest test suite"
	@echo "  make evaluate      Run 9-case empirical benchmark evaluation"
	@echo "  make run-api       Start FastAPI REST server on port 8000"
	@echo "  make run-ui        Launch Streamlit intelligence dashboard on port 8501"
	@echo "  make docker-build  Build production Docker image"
	@echo "  make docker-up     Start API and UI containers using docker-compose"
	@echo "  make docker-down   Stop and remove docker-compose containers"
	@echo "  make clean         Remove build artifacts and temporary cache directories"

install:
	pip install --upgrade pip
	pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest -v --durations=10

evaluate:
	python scripts/evaluate.py --mock

run-api:
	uvicorn api.routes:app --reload --host 0.0.0.0 --port 8000

run-ui:
	streamlit run app.py --server.port 8501

docker-build:
	docker build -t autospec-ai:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
