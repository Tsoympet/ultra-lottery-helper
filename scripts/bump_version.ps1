
param(
  [string]$IssPath = "..\ultra_lottery_helper.iss",
  [string]$Tag = "",
  [string]$Version = ""
)
if (-not (Test-Path $IssPath)) { throw "ISS not found: $IssPath" }
$ver = if ($Version) { $Version } else { $Tag.TrimStart("v") }
if ($ver -notmatch '^\d+\.\d+\.\d+$') { throw "Version must be X.Y.Z (or tag vX.Y.Z)" }
$iss = Get-Content $IssPath -Raw
if ($iss -match '(?m)^AppVersion=.*$') {
  $iss = [regex]::Replace($iss, '(?m)^AppVersion=.*$', "AppVersion=$ver")
} else {
  $iss = $iss -replace '\[Setup\]', "[Setup]`nAppVersion=$ver"
}
Set-Content -Path $IssPath -Value $iss -Encoding UTF8
Write-Host "AppVersion set to $ver" -ForegroundColor Green
