.PHONY: up down test train frontend-build demo

PYTHON ?= python3

up:
	docker compose up --build

down:
	docker compose down

test:
	$(PYTHON) -m pytest

train:
	$(PYTHON) backend/train_model.py

frontend-build:
	cd frontend && npm ci && npm run build

demo:
	$(PYTHON) scripts/demo_requests.py
