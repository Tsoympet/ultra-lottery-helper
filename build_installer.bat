@echo off
setlocal

echo === Ultra Lottery Helper - Local Build Script (Native Desktop) ===

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist dist_installer rmdir /s /q dist_installer

REM Ensure Python and pip are available
python --version || (echo Python not found! & exit /b 1)

REM Optional: install dependencies
if exist requirements.txt (
  echo Installing Python dependencies from requirements.txt...
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
)

REM Install PyInstaller if missing
pyinstaller --version >nul 2>&1 || python -m pip install pyinstaller

REM Build portable EXE with PyInstaller (native desktop entry point)
echo [1/2] Building portable EXE with PyInstaller...
set ICON_FLAG=
if exist assets\icon.ico set ICON_FLAG=--icon=assets\icon.ico
pyinstaller --onefile --noconsole %ICON_FLAG% ^
  --name ultra_lottery_helper ^
  --add-data "assets;assets" ^
  src\ulh_desktop.py
if errorlevel 1 (
    echo PyInstaller build failed!
    exit /b 1
)

REM Locate Inno Setup compiler
set "ISCC_PATH=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC_PATH%" set "ISCC_PATH=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC_PATH%" (
    echo Inno Setup compiler not found. Please install Inno Setup 6 and re-run.
    echo Download: https://jrsoftware.org/isinfo.php
    exit /b 1
)

REM Stamp version from tag if present (optional)
for /f "tokens=3 delims=/" %%a in ("%GITHUB_REF%") do set GIT_TAG=%%a
if not "%GIT_TAG%"=="" (
  echo #define MyAppVersion "%GIT_TAG%" > version.iss
  type ultra_lottery_helper.iss >> version.iss
  move /y version.iss ultra_lottery_helper.iss >nul
)

REM Build installer with Inno Setup
echo [2/2] Compiling installer with Inno Setup...
"%ISCC_PATH%" ultra_lottery_helper.iss
if errorlevel 1 (
    echo Inno Setup build failed!
    exit /b 1
)

echo === Build finished successfully! ===
echo Portable EXE: dist\ultra_lottery_helper.exe
echo Installer:    dist_installer\UltraLotteryHelperInstaller_*.exe

endlocal
pause
