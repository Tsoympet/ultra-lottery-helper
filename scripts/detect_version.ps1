
param(
  [string]$PyProject = "..\pyproject.toml",
  [string]$FallbackPy = "..\src\ultra_lottery_helper.py",
  [string]$Tag = ""
)
function Write-Version([string]$v) {
  if ($v) {
    Write-Host $v
    Set-Content -Path version_detected.txt -Value $v -Encoding UTF8
    exit 0
  } else { exit 1 }
}
if (Test-Path $PyProject) {
  $txt = Get-Content $PyProject -Raw
  if ($txt -match '(?m)^\s*version\s*=\s*\"([0-9]+\.[0-9]+\.[0-9]+)\"') { Write-Version $Matches[1] }
}
if (Test-Path $FallbackPy) {
  $src = Get-Content $FallbackPy -Raw
  if ($src -match '__version__\s*=\s*\"([0-9]+\.[0-9]+\.[0-9]+)\"') { Write-Version $Matches[1] }
  if ($src -match 'Offline Suite\s*\(v([0-9]+\.[0-9]+\.[0-9]+)\)') { Write-Version $Matches[1] }
}
if ($Tag) {
  $v = $Tag.TrimStart("v")
  if ($v -match '^[0-9]+\.[0-9]+\.[0-9]+$') { Write-Version $v }
}
Write-Error "Version not detected"; exit 2
