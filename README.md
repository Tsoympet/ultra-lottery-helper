# Ultra Lottery Helper
[English](README.md) | [Ελληνικά](README.el.md)

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](#)
[![Windows_Installer](https://img.shields.io/badge/Windows-Installer-blue)](#)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](#)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#)

**Ultra Lottery Helper** — offline-first analysis & column generation for **TZOKER / LOTTO / EuroJackpot**.
It blends **EWMA/BMA**, adaptive **luck/unluck**, **constraints**, **Gumbel Top‑k** sampling, **DPP** portfolio selection, **Monte Carlo** risk, and optional **ML** (Prophet, LightGBM, RF, XGBoost, SVM). Launches a **Gradio** UI.

> Randomness is the sea; we chart the waves. Play responsibly.

## Features
- Offline by default; optional online fetch
- Auto-merge histories from `data/history/<game>/` (CSV/XLS/XLSX)
- Diagnostics: frequency, recency, last‑digits, pairs heatmap, odd/even
- Modeling: EWMA, BMA, adaptive luck/unluck, optional ML
- Constraints: sums, odd/even, lows, consecutive, last-digit caps (adaptive/manual)
- Sampler: Gumbel Top‑k (+ optional wheels)
- Portfolio: DPP/Greedy + coverage boost; **Monte Carlo** risk
- Optional **EV re‑rank** (cost‑aware)
- Exports CSV/PNG to `exports/<game>/`
- **Plot caching** & **debounced** heavy sliders
- Windows **Installer** (icon, shortcuts, **Uninstall**)
- CI/CD: GitHub Actions (portable EXE + Installer)

## Quick Start
### Windows (Installer)
1. From Releases, download `UltraLotteryHelperInstaller_X.Y.Z.exe`
2. Install → Start Menu/Desktop shortcuts, `data/history/*` & `exports/*` folders
3. Launch → auto‑opens `http://127.0.0.1:7860`

### Portable
- Download `ultra_lottery_helper.exe`, place `data/` next to it (optional), double‑click

### Dev (Python)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
pip install -r requirements.txt
python src/ultra_lottery_helper.py
```

## Data Layout
```
data/history/{tzoker,lotto,eurojackpot}
exports/{tzoker,lotto,eurojackpot}
assets/icon.ico
```

## Local Build (Windows)
Run:
```
build_installer.bat
```
Produces:
- `dist\ultra_lottery_helper.exe` (portable)
- `dist_installer\UltraLotteryHelperInstaller_*.exe`

## CI/CD
- `.github/workflows/ci.yaml` — Linux CI (install deps, smoke compile)
- `.github/workflows/build-windows-installer.yaml` — Windows build:
  - PyInstaller (with icon)
  - Install Inno Setup via Chocolatey
  - Stamp version from release tag
  - Compile installer & upload artifacts

## License
MIT (see `LICENSE.txt`). Use responsibly; no guarantees.
