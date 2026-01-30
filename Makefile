.PHONY: install dev lint format clean test

install:
	pip install -e .

dev:
	pip install -r requirements-dev.txt
	pre-commit install

lint:
	ruff check .
	mypy .

format:
	ruff format .

test:
	pytest

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
