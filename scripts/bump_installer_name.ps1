
param(
  [string]$IssPath = "..\ultra_lottery_helper.iss",
  [string]$Tag = "",
  [string]$BuildNumber = ""
)
if (-not (Test-Path $IssPath)) { throw "ISS not found: $IssPath" }
$ver = $Tag.TrimStart("v")
if ($ver -notmatch '^\d+\.\d+\.\d+$') { throw "Tag must look like vX.Y.Z" }
if (-not $BuildNumber) { $BuildNumber = "1" }
try { $bnInt = [int]$BuildNumber; $BuildNumber = ("{0:d4}" -f $bnInt) } catch {}
$baseName = "OracleLotteryPredictor-Setup-v$ver-b$BuildNumber"
$iss = Get-Content $IssPath -Raw
if ($iss -match '(?m)^OutputBaseFilename=.*$') {
  $iss = [regex]::Replace($iss, '(?m)^OutputBaseFilename=.*$', "OutputBaseFilename=$baseName")
} else {
  $iss = $iss -replace '\[Setup\]', "[Setup]`nOutputBaseFilename=$baseName"
}
Set-Content -Path $IssPath -Value $iss -Encoding UTF8
Write-Host "OutputBaseFilename set to $baseName" -ForegroundColor Green
