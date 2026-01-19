# Development Setup Guide

This guide will help you set up the Oracle Lottery Predictor for development.

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Tsoympet/ultra-lottery-helper.git
cd ultra-lottery-helper
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/Mac:
source .venv/bin/activate
```

### 3. Install Dependencies

#### Core Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Development Dependencies
```bash
pip install -r requirements-dev.txt
```

Or install the package in editable mode (includes all dependencies):
```bash
pip install -e .
```

#### Optional: ML Dependencies
The ML libraries (scikit-learn, lightgbm, xgboost, prophet) are included in `requirements.txt` and are installed by default. They enable advanced prediction features but the application can run without them.

### 4. Install Qt/GL Runtime Libraries (Linux only)

If you're on Linux and plan to run the desktop UI, install these system libraries:

```bash
sudo apt-get update
sudo apt-get install -y \
  libegl1 libgl1 libopengl0 libglx0 \
  libxkbcommon-x11-0 libxcb-xinerama0 libxcb1 libx11-xcb1 libx11-6 \
  libdbus-1-3 libxi6 libxcursor1
```

## Running the Application

### Desktop UI

```bash
# Using the installed command (after pip install -e .)
oracle-lottery

# Or run directly
python src/ulh_desktop.py
```

### Command-Line Tools

```bash
# Learning system CLI
oracle-lottery-learn --help

# Data fetcher
python src/lottery_data_fetcher.py --help

# Scheduler
python src/lottery_scheduler.py --help

# Prediction tracker
python src/prediction_tracker.py --help
```

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html
```

On Linux, tests require the QT_QPA_PLATFORM environment variable:
```bash
QT_QPA_PLATFORM=offscreen pytest
```

## Development Tools

### Linting and Formatting

```bash
# Install dev tools
pip install black flake8 mypy isort bandit safety

# Format code
black src/

# Sort imports
isort src/

# Type checking
mypy src/

# Security scan
bandit -r src/
safety check
```

## Project Structure

```
ultra-lottery-helper/
β"œβ"€β"€ src/                      # Source code
β"‚   β"œβ"€β"€ ultra_lottery_helper.py  # Core prediction engine
β"‚   β"œβ"€β"€ ulh_desktop.py          # Desktop UI
β"‚   β"œβ"€β"€ ulh_learning.py         # Learning system
β"‚   β"œβ"€β"€ ulh_learn_cli.py        # Learning CLI
β"‚   β"œβ"€β"€ lottery_data_fetcher.py # Data fetching
β"‚   β"œβ"€β"€ lottery_scheduler.py    # Automated scheduling
β"‚   └── prediction_tracker.py   # Prediction tracking
β"œβ"€β"€ tests/                   # Test suite
β"œβ"€β"€ data/                    # Data directory
β"‚   └── history/            # Historical lottery data
β"œβ"€β"€ exports/                 # Generated predictions
β"œβ"€β"€ assets/                  # Icons and images
β"œβ"€β"€ requirements.txt         # Python dependencies
β"œβ"€β"€ requirements-dev.txt     # Development dependencies
β"œβ"€β"€ pyproject.toml           # Project configuration
└── README.md                # Main documentation
```

## Building for Distribution

### Windows Installer

```bash
# Run the build script
build_installer.bat
```

This will create:
- `dist\ultra_lottery_helper.exe` (portable executable)
- `dist_installer\OracleLotteryPredictorInstaller_*.exe` (installer)

### Manual Build with PyInstaller

```bash
pip install pyinstaller

pyinstaller --onefile --noconsole \
  --icon=assets/icon.ico \
  --name ultra_lottery_helper \
  --add-data "assets;assets" \
  src/ulh_desktop.py
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, ensure:
1. You're in the project root directory
2. You've installed the package with `pip install -e .`
3. Your virtual environment is activated

### Qt Platform Plugin Errors

On headless systems (no display), set:
```bash
export QT_QPA_PLATFORM=offscreen
```

### Missing Dependencies

Reinstall all dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Run linters: `black src/ && flake8 src/`
6. Submit a pull request

## License

MIT License - see [LICENSE.txt](LICENSE.txt) for details.
