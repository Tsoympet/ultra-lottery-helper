
param(
  [string]$IssPath = "..\ultra_lottery_helper.iss",
  [string]$Tag = "",
  [string]$Version = ""
)
if (-not (Test-Path $IssPath)) { throw "ISS not found: $IssPath" }
if (-not $Tag) { throw "Provide -Tag like v6.3.2" }
# Strip leading 'v' if present
$ver = if ($Version) { $Version } else { $Tag.TrimStart("v") }
# Validate semver-ish
if ($ver -notmatch '^\d+\.\d+\.\d+$') { throw "Tag must be like vX.Y.Z" }

$iss = Get-Content $IssPath -Raw
if ($iss -match '(?m)^AppVersion=.*$') {
  $iss = [regex]::Replace($iss, '(?m)^AppVersion=.*$', "AppVersion=$ver")
} else {
  $iss = $iss -replace '\[Setup\]', "[Setup]`nAppVersion=$ver"
}
Set-Content -Path $IssPath -Value $iss -Encoding UTF8
Write-Host "AppVersion set to $ver in $IssPath" -ForegroundColor Green
