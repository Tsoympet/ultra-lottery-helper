.PHONY: help install install-dev test lint format clean run build

help:
	@echo "Oracle Lottery Predictor - Development Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install package and dependencies"
	@echo "  install-dev  - Install package with development dependencies"
	@echo "  test         - Run test suite"
	@echo "  lint         - Run linters (flake8, mypy, bandit)"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Remove build artifacts and cache files"
	@echo "  run          - Run the desktop application"
	@echo "  run-cli      - Show available CLI commands"
	@echo ""

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

test:
	QT_QPA_PLATFORM=offscreen pytest -v

test-coverage:
	QT_QPA_PLATFORM=offscreen pytest --cov=src --cov-report=html --cov-report=term

lint:
	@echo "Running flake8..."
	flake8 src/ --max-line-length=100 --ignore=E203,W503 || true
	@echo ""
	@echo "Running mypy..."
	mypy src/ || true
	@echo ""
	@echo "Running bandit..."
	bandit -r src/ || true

format:
	@echo "Formatting with black..."
	black src/ tests/
	@echo ""
	@echo "Sorting imports with isort..."
	isort src/ tests/

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ dist_installer/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf htmlcov/ .coverage
	@echo "Clean complete!"

run:
	@echo "Starting Oracle Lottery Predictor..."
	python src/ulh_desktop.py

run-cli:
	@echo "Available CLI commands:"
	@echo ""
	@echo "  oracle-lottery          - Desktop UI"
	@echo "  oracle-lottery-learn    - Learning system CLI"
	@echo ""
	@echo "Examples:"
	@echo "  python src/lottery_data_fetcher.py --help"
	@echo "  python src/lottery_scheduler.py --help"
	@echo "  python src/prediction_tracker.py --help"
	@echo ""
