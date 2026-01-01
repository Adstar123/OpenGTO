@echo off
REM OpenGTO Build Script - Windows Batch Wrapper
REM This calls the PowerShell build script

echo ========================================
echo    OpenGTO Build Script
echo ========================================
echo.

REM Check if PowerShell is available
where powershell >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: PowerShell not found. Please install PowerShell.
    exit /b 1
)

REM Run the PowerShell script with execution policy bypass
powershell -ExecutionPolicy Bypass -File "%~dp0build.ps1" %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo Build failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Build completed successfully!
pause
