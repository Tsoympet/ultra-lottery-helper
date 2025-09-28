
# Code Signing (Windows)

## Requirements
- Windows 10/11
- Python 3.10+
- `pip install -r requirements.txt`
- PyInstaller: `pip install pyinstaller`
- Inno Setup (ISCC in PATH)
- SignTool (μέρος του Windows SDK)
- Code Signing Certificate (OV ή EV). Για EV απαιτείται το USB token + drivers.

## Γρήγορη χρήση
Από PowerShell, μέσα στο `scripts/`:
```powershell
# Build, sign binaries, build & sign installer
.\build_and_sign.ps1 -CertThumbprint "YOUR_CERT_THUMBPRINT"
# ή χωρίς thumbprint (auto-select κατάλληλο cert)
.\build_and_sign.ps1
```

Μόνο υπογραφή υπαρχόντων binaries:
```powershell
.\sign-all.ps1 -Folder "..\dist\OracleLotteryPredictor" -CertThumbprint "YOUR_CERT_THUMBPRINT"
```

## Σημειώσεις
- Χρησιμοποιείται timestamp server: `http://timestamp.digicert.com` (αλλάζεις με παράμετρο).
- Αν έχεις EV token, φρόντισε να φαίνεται στα Personal Certificates.
- Το Inno Setup υπογράφει **installer+uninstaller** αυτόματα (βλέπε `.iss`).
