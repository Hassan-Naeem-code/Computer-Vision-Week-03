.PHONY: help install install-dev train test lint format clean

help:
	@echo "Urban Scene CNN - Available commands"
	@echo "====================================="
	@echo "make install          Install dependencies"
	@echo "make install-dev      Install development dependencies"
	@echo "make train            Train the model"
	@echo "make test             Run tests"
	@echo "make lint             Run linting checks"
	@echo "make format           Format code with black"
	@echo "make clean            Remove generated files"
	@echo "make requirements     Update requirements files"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install -r requirements-dev.txt

train:
	python -m src.main --config configs/default.yaml

train-custom:
	python -m src.main --config $(CONFIG)

test:
	pytest tests/ -v --cov=src

lint:
	flake8 src/ tests/
	pylint src/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	rm -rf build/ dist/ *.egg-info

requirements:
	pip freeze > requirements.txt

.DEFAULT_GOAL := help
