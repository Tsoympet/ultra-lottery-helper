# CI/CD for Oracle Lottery Predictor

## Trigger
Create a tag `vX.Y.Z`:
```bash
git tag v6.3.6
git push origin v6.3.6
```

## Jobs
- **build-windows** (GitHub runner): PyInstaller unsigned build → uploads `unsigned-dist/`.
- **sign-and-installer** (self-hosted Windows): detects version, bumps AppVersion & installer filename, signs binaries, builds & signs installer, uploads `signing-log`, optional `virustotal-response`, and `installer`.
- **release** (Ubuntu): creates GitHub Release with installer and unsigned-dist; appends CHANGELOG.md to the body if present.

## Secrets
- `CERT_THUMBPRINT` (optional): SHA1 thumbprint. If empty, SignTool uses `/a`.
- `TIMESTAMP_URL` (optional): e.g. `http://timestamp.digicert.com`
- `VIRUSTOTAL_API_KEY` (optional): enables VirusTotal upload.

## Requirements (self-hosted Windows runner)
- Labels: `self-hosted, windows, ev-token`
- In PATH: **signtool** (Windows SDK), **ISCC.exe** (Inno Setup), **python** + **pyinstaller**
- Access to your EV/OV code-signing certificate/token.
