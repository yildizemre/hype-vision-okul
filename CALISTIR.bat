@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Hype Vision
set PORT=8000

rem Kurulum yapilmamissa once BASLAT.bat calismali
if not exist venv\Scripts\python.exe (
  echo Ilk kurulum yapilmamis, BASLAT.bat calistiriliyor...
  call BASLAT.bat
  exit /b
)

rem Zaten calisiyorsa sadece paneli ac
netstat -ano | findstr ":%PORT% " | findstr LISTENING >nul
if not errorlevel 1 (
  echo Hype Vision zaten calisiyor.
  start "" http://localhost:%PORT%
  exit /b
)

echo Hype Vision baslatiliyor...
start "Hype Vision Sunucu" /min cmd /c "venv\Scripts\python.exe app.py > server.log 2>&1"

rem Sunucu hazir olana kadar bekle (en fazla ~60 sn)
for /l %%i in (1,1,60) do (
  netstat -ano | findstr ":%PORT% " | findstr LISTENING >nul && goto hazir
  ping -n 2 127.0.0.1 >nul
)
echo Sunucu baslamadi. Ayrintilar icin server.log dosyasina bakin.
pause
exit /b

:hazir
start "" http://localhost:%PORT%
echo Hazir: http://localhost:%PORT%
ping -n 3 127.0.0.1 >nul
