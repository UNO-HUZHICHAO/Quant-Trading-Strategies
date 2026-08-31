param(
    [string]$XeLaTeX = "xelatex"
)

$ErrorActionPreference = "Stop"
$srcDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $srcDir
$reportDir = Join-Path $srcDir "report"
$buildDir = Join-Path ([System.IO.Path]::GetTempPath()) ("intraday-factor-report-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $buildDir | Out-Null

Push-Location $reportDir
try {
    & $XeLaTeX -interaction=nonstopmode -halt-on-error -jobname=report "-output-directory=$buildDir" .\report.tex
    & $XeLaTeX -interaction=nonstopmode -halt-on-error -jobname=report "-output-directory=$buildDir" .\report.tex
    Copy-Item -LiteralPath (Join-Path $buildDir "report.pdf") -Destination (Join-Path $projectDir "report.pdf") -Force
}
finally {
    Pop-Location
    Remove-Item -LiteralPath $buildDir -Recurse -Force
}

