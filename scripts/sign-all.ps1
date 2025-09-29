
param(
    [string]$Folder = "..\dist\OracleLotteryPredictor",
    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [string]$CertThumbprint = ""
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $Folder)) { throw "Folder not found: $Folder" }
$files = Get-ChildItem $Folder -Recurse -Include *.exe,*.dll
foreach ($f in $files) {
    $cmd = @("sign", "/fd", "sha256")
    if ([string]::IsNullOrWhiteSpace($CertThumbprint)) { $cmd += "/a" } else { $cmd += @("/sha1", $CertThumbprint) }
    $cmd += @("/tr", $TimestampUrl, "/td", "sha256", $f.FullName)
    & signtool @cmd
}
Write-Host "Signing completed." -ForegroundColor Green
