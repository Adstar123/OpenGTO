# OpenGTO Build Script for Windows
# This script builds both the Python backend and Electron frontend

param(
    [switch]$SkipPython,
    [switch]$SkipFrontend,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   OpenGTO Build Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ProjectRoot = $PSScriptRoot
$FrontendDir = Join-Path $ProjectRoot "frontend"
$DistPythonDir = Join-Path $ProjectRoot "dist-python"

# Clean build directories if requested
if ($Clean) {
    Write-Host "Cleaning build directories..." -ForegroundColor Yellow

    if (Test-Path $DistPythonDir) {
        Remove-Item -Recurse -Force $DistPythonDir
    }

    $ElectronBuilderDist = Join-Path $FrontendDir "dist-electron-builder"
    if (Test-Path $ElectronBuilderDist) {
        Remove-Item -Recurse -Force $ElectronBuilderDist
    }

    Write-Host "Clean complete!" -ForegroundColor Green
    Write-Host ""
}

# Step 1: Build Python backend with PyInstaller
if (-not $SkipPython) {
    Write-Host "Step 1: Building Python backend..." -ForegroundColor Cyan
    Write-Host "--------------------------------------" -ForegroundColor Gray

    # Check if PyInstaller is installed
    $pyinstallerCheck = python -c "import PyInstaller" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing PyInstaller..." -ForegroundColor Yellow
        pip install pyinstaller
    }

    # Create output directory
    if (-not (Test-Path $DistPythonDir)) {
        New-Item -ItemType Directory -Path $DistPythonDir | Out-Null
    }

    # Run PyInstaller
    Write-Host "Running PyInstaller..." -ForegroundColor Yellow
    Push-Location $ProjectRoot

    pyinstaller --distpath $DistPythonDir --workpath (Join-Path $ProjectRoot "build-python") --clean opengto_backend.spec

    if ($LASTEXITCODE -ne 0) {
        Write-Host "PyInstaller build failed!" -ForegroundColor Red
        Pop-Location
        exit 1
    }

    Pop-Location

    # Verify the executable was created
    $ExePath = Join-Path $DistPythonDir "opengto_backend.exe"
    if (Test-Path $ExePath) {
        $ExeSize = (Get-Item $ExePath).Length / 1MB
        Write-Host "Python backend built successfully!" -ForegroundColor Green
        Write-Host "  Executable: $ExePath" -ForegroundColor Gray
        Write-Host "  Size: $([math]::Round($ExeSize, 2)) MB" -ForegroundColor Gray
    } else {
        Write-Host "Error: Executable not found at $ExePath" -ForegroundColor Red
        exit 1
    }

    Write-Host ""
}

# Step 2: Build Electron frontend
if (-not $SkipFrontend) {
    Write-Host "Step 2: Building Electron frontend..." -ForegroundColor Cyan
    Write-Host "--------------------------------------" -ForegroundColor Gray

    Push-Location $FrontendDir

    # Install dependencies if needed
    if (-not (Test-Path (Join-Path $FrontendDir "node_modules"))) {
        Write-Host "Installing npm dependencies..." -ForegroundColor Yellow
        npm install
    }

    # Build the application
    Write-Host "Building application..." -ForegroundColor Yellow
    npm run build

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Electron build failed!" -ForegroundColor Red
        Pop-Location
        exit 1
    }

    Pop-Location

    Write-Host "Electron frontend built successfully!" -ForegroundColor Green
    Write-Host ""
}

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Build Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$InstallerDir = Join-Path $FrontendDir "dist-electron-builder"
if (Test-Path $InstallerDir) {
    Write-Host "Installer location:" -ForegroundColor Yellow
    Get-ChildItem -Path $InstallerDir -Filter "*.exe" | ForEach-Object {
        Write-Host "  $($_.FullName)" -ForegroundColor Gray
        $Size = $_.Length / 1MB
        Write-Host "  Size: $([math]::Round($Size, 2)) MB" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "To test the installer, run one of the .exe files in:" -ForegroundColor Yellow
Write-Host "  $InstallerDir" -ForegroundColor Gray
