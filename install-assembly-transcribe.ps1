param(
    [string]$RepoRoot = (Split-Path -Parent $MyInvocation.MyCommand.Path),
    [string]$VenvName = '.venv',
    [string]$ShimDir = (Join-Path $env:USERPROFILE 'bin')
)

$ErrorActionPreference = 'Stop'

function Resolve-PythonCommand {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @{ Path = $python.Source; UseLauncher = $false }
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return @{ Path = $py.Source; UseLauncher = $true }
    }

    throw 'Python not found on PATH. Install Python 3.9+ and retry.'
}

function Ensure-UserPathIncludes {
    param([Parameter(Mandatory=$true)][string]$PathToAdd)

    if (-not $PathToAdd) { return }
    if (-not (Test-Path $PathToAdd)) { return }

    $currentUser = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not $currentUser) { $currentUser = '' }

    $parts = $currentUser -split ';' | Where-Object { $_ -and $_.Trim() }
    if ($parts -notcontains $PathToAdd) {
        $newUser = ($parts + $PathToAdd) -join ';'
        [Environment]::SetEnvironmentVariable('Path', $newUser, 'User')
    }

    $procParts = $env:Path -split ';' | Where-Object { $_ -and $_.Trim() }
    if ($procParts -notcontains $PathToAdd) {
        $env:Path = ($procParts + $PathToAdd) -join ';'
    }
}

$RepoRoot = (Resolve-Path $RepoRoot).Path
$requirementsPath = Join-Path $RepoRoot 'requirements.txt'
$scriptPath = Join-Path $RepoRoot 'assembly_transcribe.py'
$venvPath = Join-Path $RepoRoot $VenvName
$venvPython = Join-Path $venvPath 'Scripts\python.exe'

if (-not (Test-Path $requirementsPath)) {
    throw "requirements.txt not found at $requirementsPath"
}
if (-not (Test-Path $scriptPath)) {
    throw "assembly_transcribe.py not found at $scriptPath"
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment at $venvPath"
    $pythonCmd = Resolve-PythonCommand
    if ($pythonCmd.UseLauncher) {
        & $pythonCmd.Path -3 -m venv $venvPath
    } else {
        & $pythonCmd.Path -m venv $venvPath
    }
}

Write-Host 'Installing/updating dependencies...'
& $venvPython -m pip install -r $requirementsPath

New-Item -ItemType Directory -Force -Path $ShimDir | Out-Null

$psShimPath = Join-Path $ShimDir 'assembly_transcribe.ps1'
$cmdShimPath = Join-Path $ShimDir 'assembly_transcribe.cmd'

$psShim = @'
param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
$pythonExe = "__PYTHON__"
$scriptPath = "__SCRIPT__"
& $pythonExe $scriptPath @Args
exit $LASTEXITCODE
'@

$cmdShim = @'
@echo off
setlocal
set "PY_EXE=__PYTHON__"
set "SCRIPT=__SCRIPT__"
"%PY_EXE%" "%SCRIPT%" %*
exit /b %ERRORLEVEL%
'@

$psShim = $psShim.Replace('__PYTHON__', $venvPython).Replace('__SCRIPT__', $scriptPath)
$cmdShim = $cmdShim.Replace('__PYTHON__', $venvPython).Replace('__SCRIPT__', $scriptPath)

Set-Content -Path $psShimPath -Value $psShim -Encoding ASCII
Set-Content -Path $cmdShimPath -Value $cmdShim -Encoding ASCII

Ensure-UserPathIncludes -PathToAdd $ShimDir

Write-Host 'Installed assembly_transcribe shim.'
Write-Host "Repo: $RepoRoot"
Write-Host "Venv: $venvPath"
Write-Host "Shim: $cmdShimPath"
Write-Host 'Open a new PowerShell window or run `refreshenv` if needed.'
