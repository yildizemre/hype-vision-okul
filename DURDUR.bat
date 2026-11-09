@echo off
chcp 65001 >nul
set PORT=8000
set FOUND=0
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT% " ^| findstr LISTENING') do (
  taskkill /PID %%p /F >nul 2>&1
  set FOUND=1
)
if "%FOUND%"=="1" (echo Hype Vision durduruldu.) else (echo Hype Vision zaten calismiyor.)
ping -n 3 127.0.0.1 >nul
