@echo off
setlocal
title MCAI - Watch Training
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\watch-windows.ps1"
if errorlevel 1 pause
