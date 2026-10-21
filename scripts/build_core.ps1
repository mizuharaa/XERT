<#
.SYNOPSIS
From the checkout root: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_core.ps1
Requires the existing .venv runtime dependencies, CMake >= 3.20, and VS 2022 C++ Build Tools.
Builds all four targets; CTest exports its own bench fixture. Windows only.
#>
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Push-Location $repo
try {
    $python = Join-Path $repo '.venv\Scripts\python.exe'
    & $python -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('pybind11') is None)"
    if ($LASTEXITCODE -ne 0) {
        & $python -m ensurepip
        if ($LASTEXITCODE -ne 0) { throw 'ensurepip failed' }
        & $python -m pip install pybind11==3.1.0
        if ($LASTEXITCODE -ne 0) { throw 'pybind11 installation failed' }
    }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $installation = & $vswhere -latest -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (!$installation) { throw 'VS 2022 C++ Build Tools not found' }
    $vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
    $env:PATH = (Split-Path $vswhere -Parent) + ';' + $env:PATH
    # cmd stays attached to these pipes; no Start-Process or visible window.
    & $env:ComSpec /d /c "call `"$vcvars`" >nul && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
    }
    if ($LASTEXITCODE -ne 0) { throw 'vcvars64 failed' }
    & cmake --preset windows
    if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed' }
    & cmake --build --preset windows
    if ($LASTEXITCODE -ne 0) { throw 'CMake build failed' }
} finally {
    Pop-Location
}
