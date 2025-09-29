
param(
    [string]$Python = "py",
    [string]$Name = "OracleLotteryPredictor",
    [string]$Icon = "..\assets\icon.ico",
    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [string]$CertThumbprint = "",
    [switch]$EV,
    [switch]$SkipInstallerBuild
)

$ErrorActionPreference = "Stop"
Write-Host "=== Cleaning old build folders ===" -ForegroundColor Cyan
Remove-Item -Recurse -Force ..\build, ..\dist -ErrorAction SilentlyContinue
Write-Host "=== Building with PyInstaller ===" -ForegroundColor Cyan

pyinstaller ..\src\ulh_desktop.py `
  --name $Name `
  --noconsole `
  --icon $Icon `
  --add-data "..\src;src" `
  --add-data "..\assets;assets" `
  --add-data "..\data;data"

$dist = Join-Path (Resolve-Path "..") "dist\$Name"
if (-not (Test-Path $dist)) {
    throw "Build output not found: $dist"
}

Write-Host "=== Signing PyInstaller binaries ===" -ForegroundColor Cyan
$log = Join-Path (Resolve-Path "..") "signing-log.txt"
$files = Get-ChildItem $dist -Recurse -Include *.exe,*.dll

foreach ($f in $files) {
    $cmd = @("sign", "/fd", "sha256")
    if ([string]::IsNullOrWhiteSpace($CertThumbprint)) {
        $cmd += "/a"
    } else {
        $cmd += @("/sha1", $CertThumbprint)
    }
    $cmd += @("/tr", $TimestampUrl, "/td", "sha256", $f.FullName)
    & signtool @cmd 2>&1 | Tee-Object -FilePath $log -Append
}

Write-Host "=== Verifying primary EXE ===" -ForegroundColor Cyan
$primary = Join-Path $dist "$Name.exe"
if (-not (Test-Path $primary)) {
    $fallbacks = @(
        (Join-Path $dist "ultra_lottery_helper.exe"),
        (Join-Path $dist "UltraLotteryHelper.exe"),
        (Join-Path $dist "ulh_desktop.exe")
    )
    foreach ($f in $fallbacks) {
        if (Test-Path $f) { $primary = $f; break }
    }
}
if (-not (Test-Path $primary)) {
    Write-Host "Listing dist for diagnostics:" -ForegroundColor Yellow
    Get-ChildItem $dist -Recurse | Select-Object FullName,Length | Format-Table | Out-Host
    throw "PyInstaller output missing (expected $Name.exe in $dist)"
}
& signtool verify /pa /all $primary 2>&1 | Tee-Object -FilePath $log -Append | Out-Host

if ($SkipInstallerBuild) { exit 0 }

Write-Host "=== Building Inno Setup installer (auto-sign) ===" -ForegroundColor Cyan
Push-Location ..
try { iscc .\ultra_lottery_helper.iss | Out-Host } finally { Pop-Location }

$outDir = Join-Path (Resolve-Path "..") "Output"
$setupExe = Get-ChildItem $outDir -Filter *.exe | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($setupExe) {
    Write-Host "=== Post-signing installer ===" -ForegroundColor Cyan
    $cmd = @("sign", "/fd", "sha256")
    if ([string]::IsNullOrWhiteSpace($CertThumbprint)) { $cmd += "/a" } else { $cmd += @("/sha1", $CertThumbprint) }
    $cmd += @("/tr", $TimestampUrl, "/td", "sha256", $setupExe.FullName)
    & signtool @cmd
    Write-Host "Installer signed: $($setupExe.FullName)" -ForegroundColor Green
} else {
    Write-Warning "Installer not found in Output\ folder."
}
