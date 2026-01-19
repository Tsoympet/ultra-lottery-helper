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

## Features
- Offline by default; optional online fetch
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

## Quick Start
### Windows (Installer)
1. Download `OracleLotteryPredictorInstaller_X.Y.Z.exe` from **Releases**.
2. Run installer β†’ Start Menu/Desktop shortcuts are created; app data folders (`data/history/*`, `exports/*`) are set up.
3. Launch **Oracle Lottery Predictor (Desktop)** from Start Menu or Desktop. (No browser needed.)

### Portable
- Download `ultra_lottery_helper.exe`, optionally place a `data/` folder next to it, then double-click to run.

### Dev (Python)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
python src/ulh_desktop.py
```

## Data Layout
```
data/history/{tzoker,lotto,eurojackpot,uk_national_lottery,la_primitiva,
              superenalotto,loto_france,lotto_6aus49,austrian_lotto,swiss_lotto}
exports/{tzoker,lotto,eurojackpot,uk_national_lottery,la_primitiva,
         superenalotto,loto_france,lotto_6aus49,austrian_lotto,swiss_lotto}
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
