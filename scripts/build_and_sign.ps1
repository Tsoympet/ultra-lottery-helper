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
$files = Get-ChildItem $dist -Recurse -Include *.exe,*.dll

foreach ($f in $files) {
    $cmd = @("sign", "/fd", "sha256")
    if ([string]::IsNullOrWhiteSpace($CertThumbprint)) {
        $cmd += "/a"
    } else {
        $cmd += @("/sha1", $CertThumbprint)
    }
    $cmd += @("/tr", $TimestampUrl, "/td", "sha256", $f.FullName)
    & signtool @cmd
}

Write-Host "=== Verifying primary EXE ===" -ForegroundColor Cyan
& signtool verify /pa /all (Join-Path $dist "$Name.exe") | Out-Host

if ($SkipInstallerBuild) {
    Write-Host "Skipping installer build as requested."
    exit 0
}

Write-Host "=== Building Inno Setup installer (auto-sign) ===" -ForegroundColor Cyan
Push-Location ..
try {
    & iscc .\ultra_lottery_helper.iss | Out-Host
} finally {
    Pop-Location
}

$outDir = Join-Path (Resolve-Path "..") "Output"
$setupExe = Get-ChildItem $outDir -Filter *.exe | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($setupExe) {
    Write-Host "=== Post-signing installer ===" -ForegroundColor Cyan
    $cmd = @("sign", "/fd", "sha256")
    if ([string]::IsNullOrWhiteSpace($CertThumbprint)) {
        $cmd += "/a"
    } else {
        $cmd += @("/sha1", $CertThumbprint)
    }
    $cmd += @("/tr", $TimestampUrl, "/td", "sha256", $setupExe.FullName)
    & signtool @cmd
    Write-Host "Installer signed: $($setupExe.FullName)" -ForegroundColor Green
} else {
    Write-Warning "Installer not found in Output\ folder."
}
