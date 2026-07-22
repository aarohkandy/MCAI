@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Drag your legitimate unminified Eaglercraft 1.12 u2 offline HTML onto this file.
  pause
  exit /b 1
)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0BUILD_EAGLER_CLIENT.ps1" "%~1"
if errorlevel 1 pause
