# Changelog

All notable changes to this project will be documented in this file.

## [v1.0.0] - 2025-09-07
### Added
- First public release of **Oracle Lottery Predictor**.
- Support for **TZOKER, LOTTO, EuroJackpot**.
- Unified **Gradio** UI.
- History loading from local files or optional online fetch.
- Diagnostics: frequency, recency, last-digit distribution, pairs heatmap, odd/even.
- Modeling: **EWMA**, **BMA**, adaptive **luck/unluck**, and constraints.
- Sampling: **Gumbel Top-k**, optional wheels.
- Portfolio selection: **DPP** / Greedy with Monte Carlo risk.
- Self-learning replay and walk-forward cross-validation.
- Export to CSV & PNG.
- **Windows Installer** (with icon, shortcuts, and Uninstaller).
- CI/CD via GitHub Actions (portable exe + installer builds).

### Changed
- Improved performance with plot caching and debounced heavy sliders.

### Fixed
- Minor bug fixes in constraints & export modules.

---

## [Unreleased]
- Support for additional lottery games.
- Integration of advanced ML models.
- Web dashboard (GitHub Pages) with live guides and screenshots.

