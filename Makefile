.PHONY: help install migrate lint test evaluate run-api run-web docker-build docker-up docker-down clean

help:
	@echo "AutoSpec AI v2.0 — Command Reference"
	@echo "--------------------------------------------------------"
	@echo "  make install       Install backend package + dev tools and frontend packages"
	@echo "  make migrate       Apply Alembic migrations to DATABASE_URL_UNPOOLED / local SQLite"
	@echo "  make lint          Run Ruff and frontend ESLint/TypeScript"
	@echo "  make test          Run backend pytest and frontend Vitest suites"
	@echo "  make evaluate      Run 362-case benchmark & ablation suite"
	@echo "  make run-api       Start FastAPI REST server on port 8000"
	@echo "  make run-web       Start React + Vite frontend on port 5173"
	@echo "  make docker-up     Build & start Postgres(pgvector), API, and Web containers"
	@echo "  make docker-down   Stop and remove containers"

install:
	pip install --upgrade pip
	pip install -e ".[dev]"
	cd frontend && npm install

migrate:
	alembic upgrade head

lint:
	ruff check .
	cd frontend && npm run lint && npx tsc -b --noEmit

test:
	AUTOSPEC_EMBEDDING_MODEL=mock pytest -v --durations=10
	cd frontend && npm test

evaluate:
	python scripts/evaluate.py --mock

run-api:
	uvicorn api.routes:app --reload --host 127.0.0.1 --port 8000

run-web:
	cd frontend && npm run dev

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov frontend/dist
