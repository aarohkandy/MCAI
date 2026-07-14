@echo off
setlocal
title MCAI - Stop Training
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop-windows.ps1"
echo.
pause
