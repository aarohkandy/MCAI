@echo off
setlocal
title MCAI - Setup and Training
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch-windows.ps1"
set "MCAI_EXIT=%ERRORLEVEL%"
if not "%MCAI_EXIT%"=="0" (
  echo.
  echo MCAI stopped with an error. The message above explains what needs attention.
  echo You can run this file again after fixing it; completed setup will be reused.
  pause
)
exit /b %MCAI_EXIT%
