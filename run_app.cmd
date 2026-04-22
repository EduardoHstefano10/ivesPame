@echo off
REM Lanza run_app.ps1 saltando la politica de ejecucion (para doble-click desde Windows).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_app.ps1" %*
pause
