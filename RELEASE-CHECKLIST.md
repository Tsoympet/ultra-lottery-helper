# Oracle Lottery Predictor — Release Checklist

## Pre-Build
- [ ] Update `CHANGELOG.md` with latest features/fixes
- [ ] Bump version in GitHub tag (`vX.Y.Z`)
- [ ] Verify `requirements.txt` is correct (no unused deps)

## Assets
- [ ] Ensure `assets/icon.ico` exists and is up-to-date
- [ ] Verify installer branding (icon, name, version)

## Data
- [ ] Ensure `data/history/tzoker/*` is updated
- [ ] Ensure `data/history/lotto/*` is updated
- [ ] Ensure `data/history/eurojackpot/*` is updated
- [ ] Ensure export folders (`exports/*`) are clean/empty if bundling

## Build
- [ ] Run `build_installer.bat` locally (Windows)
- [ ] Verify `dist/ultra_lottery_helper.exe` runs correctly
- [ ] Verify `dist_installer/OracleLotteryPredictorInstaller_vX.Y.Z.exe` installs and uninstalls cleanly

## Release
- [ ] Push all changes to `main`
- [ ] Create Git tag: `git tag vX.Y.Z && git push origin vX.Y.Z`
- [ ] GitHub Actions should build EXE + Installer
- [ ] Verify artifacts are attached to GitHub Release
- [ ] Double-check SHA256 checksum of installer

## Post-Release
- [ ] Announce release (README, social, etc.)
- [ ] Archive old installers if needed
