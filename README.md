# Oracle Lottery Predictor
[English](README.md) | [Ξ•Ξ»Ξ»Ξ·Ξ½ΞΉΞΊΞ¬](README.el.md)

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](#)
[![Windows_Installer](https://img.shields.io/badge/Windows-Installer-blue)](#)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](#)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#)

**Oracle Lottery Predictor** β€” offline-first analysis & column generation for **TZOKER / LOTTO / EuroJackpot**.  
Blend of **EWMA/BMA**, adaptive **luck/unluck**, **constraints**, **Gumbel Top-k** sampling, **DPP** portfolio selection, **Monte Carlo** risk, and optional **ML** (Prophet, LightGBM, RF, XGBoost, SVM).  
Now ships as a **native desktop app (PySide6)** β€” no browser, no local server.

> Randomness is the sea; we chart the waves. Play responsibly.

## Supported Lotteries

### Greek Lotteries
- **TZOKER** πŸ‡¬πŸ‡· - 5 numbers (1-45) + 1 Joker (1-20)
- **LOTTO** πŸ‡¬πŸ‡· - 6 numbers (1-49)

### Pan-European
- **EuroJackpot** πŸ‡ͺπŸ‡Ί - 5 numbers (1-50) + 2 Euro numbers (1-12)

### United Kingdom
- **UK National Lottery** πŸ‡¬πŸ‡§ - 6 numbers (1-59)

### Spain
- **La Primitiva** πŸ‡ͺπŸ‡Έ - 6 numbers (1-49) + 1 reintegro (1-10)

### Italy
- **SuperEnalotto** πŸ‡?πŸ‡Ή - 6 numbers (1-90)

### France
- **Loto** πŸ‡«πŸ‡· - 5 numbers (1-49) + 1 Chance (1-10)

### Germany
- **Lotto 6aus49** πŸ‡©πŸ‡ͺ - 6 numbers (1-49) + 1 Superzahl (1-10)

### Austria
- **Austrian Lotto** πŸ‡¦πŸ‡Ή - 6 numbers (1-45)

### Switzerland
- **Swiss Lotto** πŸ‡¨πŸ‡­ - 6 numbers (1-42) + 1 Lucky number (1-6)

**Note**: All supported lotteries offer progressive jackpots. See [LOTTERY_RESULTS_SOURCES.md](LOTTERY_RESULTS_SOURCES.md) for official results URLs and detailed information about each lottery.

## Features
- Offline by default; optional online fetch
- **βœ¨ NEW: AI/IA Learning System** - Adaptive machine learning that improves predictions over time (see [AI_SYSTEM_STATUS.md](AI_SYSTEM_STATUS.md))
- **βœ¨ NEW: Automated Live Feed** - Fetch and store latest draw results automatically (see [DATA_FETCHER_README.md](DATA_FETCHER_README.md))
- **βœ¨ NEW: Automated Scheduling** - Schedule periodic data fetching with configurable intervals (see [SCHEDULER_README.md](SCHEDULER_README.md))
- **βœ¨ NEW: Prediction Tracking** - Save predictions, compare with results, track accuracy over time (see [PREDICTION_TRACKER_README.md](PREDICTION_TRACKER_README.md))
- Auto-merge histories from `data/history/<game>/` (CSV/XLS/XLSX)
- Diagnostics: frequency, recency, last-digits, pairs heatmap, odd/even
- Modeling: EWMA, BMA, adaptive luck/unluck, optional ML
- Constraints: sums, odd/even, lows, consecutive, last-digit caps (adaptive/manual)
- Sampler: Gumbel Top-k (+ optional wheels)
- Portfolio: DPP/Greedy + coverage boost; **Monte Carlo** risk
- Optional **EV re-rank** (cost-aware)
- Exports CSV/PNG to `exports/<game>/`
- **Plot caching** & **debounced** heavy sliders
- Windows **Installer** (icon, shortcuts, **Uninstall**)
- CI/CD: GitHub Actions (portable EXE + Installer)
- **βœ¨ NEW: Docker Support** - Containerized deployment with Docker and Docker Compose (see [DOCKER_CMAKE_README.md](DOCKER_CMAKE_README.md))
- **βœ¨ NEW: CMake Build System** - Cross-platform build support with CMake (see [DOCKER_CMAKE_README.md](DOCKER_CMAKE_README.md))

## Quick Start
### Windows (Installer)
1. Download `OracleLotteryPredictorInstaller_X.Y.Z.exe` from **Releases**.
2. Run installer β†’ Start Menu/Desktop shortcuts are created; app data folders (`data/history/*`, `exports/*`) are set up.
3. Launch **Oracle Lottery Predictor (Desktop)** from Start Menu or Desktop. (No browser needed.)

### Portable
- Download `ultra_lottery_helper.exe`, optionally place a `data/` folder next to it, then double-click to run.

### Dev (Python)
For detailed development setup instructions, see [SETUP.md](SETUP.md).

Quick start:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install --upgrade pip
pip install -e .

# Run the application
oracle-lottery
# Or
python src/ulh_desktop.py
```

### Docker Deployment
Run the application in a container:

```bash
# Build and run with Docker
docker build -t oracle-lottery-predictor .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/exports:/app/exports \
  oracle-lottery-predictor python src/ulh_desktop.py

# Or use Docker Compose
docker-compose up oracle-lottery

# For detailed Docker and CMake instructions, see DOCKER_CMAKE_README.md
```

### Automated Data Fetching (New!)
```bash
# Fetch latest draw results for all lotteries
python src/lottery_data_fetcher.py --all

# Fetch specific lottery
python src/lottery_data_fetcher.py --game EUROJACKPOT

# Check fetch status
python src/lottery_data_fetcher.py --status

# See full documentation
cat DATA_FETCHER_README.md
```

### Automated Scheduling (New!)
```bash
# Enable automated fetching for all lotteries (every 12 hours)
python src/lottery_scheduler.py --enable-all

# Start the scheduler (runs continuously)
python src/lottery_scheduler.py --start

# Check scheduler status
python src/lottery_scheduler.py --status

# See full documentation
cat SCHEDULER_README.md
```

### Prediction Tracking (New!)
```bash
# Save a prediction for upcoming draw
python src/prediction_tracker.py --save EUROJACKPOT \
    --numbers "5,12,18,27,33,2,8" \
    --draw-date "2026-01-25"

# Compare predictions with actual results
python src/prediction_tracker.py --compare EUROJACKPOT \
    --draw-date "2026-01-25" \
    --actual "5,12,19,27,33,2,9"

# Auto-compare all pending predictions
python src/prediction_tracker.py --auto-compare

# View accuracy statistics
python src/prediction_tracker.py --stats

# View pending predictions
python src/prediction_tracker.py --pending

# See full documentation
cat PREDICTION_TRACKER_README.md
```

### AI/IA Learning System (New!)
```bash
# Record predictions for learning
python src/ulh_learn_cli.py record-portfolio TZOKER "1 5 12 27 38" "3 14 22 33 41"

# Record actual draw outcome
python src/ulh_learn_cli.py record-outcome TZOKER --main "3 14 22 33 41" --sec "5"

# Trigger learning and parameter optimization
python src/ulh_learn_cli.py learn TZOKER --k 100 --replay 2

# The AI system automatically adapts:
# - Ensemble weights (EWMA, recency, ML)
# - Luck/unluck factors
# - Memory half-life
# Based on actual prediction performance!

# See full documentation
cat AI_SYSTEM_STATUS.md
```

## Data Layout
```
data/history/{tzoker,lotto,eurojackpot,
              uk_national_lottery,la_primitiva,superenalotto,loto_france,lotto_6aus49,austrian_lotto,swiss_lotto,
              us_powerball,us_mega_millions,australia_powerball,canada_lotto_649,japan_loto_6,south_africa_powerball}
exports/{tzoker,lotto,eurojackpot,
         uk_national_lottery,la_primitiva,superenalotto,loto_france,lotto_6aus49,austrian_lotto,swiss_lotto,
         us_powerball,us_mega_millions,australia_powerball,canada_lotto_649,japan_loto_6,south_africa_powerball}
assets/
  β"œβ"€β"€ flags/          # Country flags for each lottery
  β"œβ"€β"€ lottery_icons/  # Official lottery icons
  └── icon.ico        # Main app icon
```

## Local Build (Windows)
Run:
```
build_installer.bat
```
Produces:
- `dist\ultra_lottery_helper.exe` (portable)
- `dist_installer\OracleLotteryPredictorInstaller_*.exe`

## CI/CD
- `.github/workflows/ci.yml` β€” CI on Linux: installs deps, headless-safe imports (Qt offscreen), optional pytest
- `.github/workflows/build-windows-installer.yml` β€” Windows build:
  - PyInstaller (**desktop entry:** `src/ulh_desktop.py`, bundles `assets`)
  - Inno Setup via Chocolatey
  - Stamps version from release tag (`vX.Y.Z`)
  - Builds installer, computes **SHA256**, uploads artifacts, auto-attaches to Release

## Requirements
- Windows 10/11 for the packaged EXE/installer
- Python 3.10+ (for dev runs)
- Optional ML libraries (enabled automatically if installed): scikit-learn, lightgbm, xgboost, prophet

## License
MIT (see `LICENSE.txt`). Use responsibly; no guarantees.
