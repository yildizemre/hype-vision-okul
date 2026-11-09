@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem Windows'a giris yapilinca Hype Vision'i otomatik baslatir ve masaustune kisayol ekler.
rem Kaldirmak icin: OTOMATIK_BASLAT.bat kaldir

if /i "%1"=="kaldir" (
  schtasks /delete /tn "HypeVision" /f >nul 2>&1
  del "%USERPROFILE%\Desktop\Hype Vision.lnk" >nul 2>&1
  echo Otomatik baslatma kaldirildi.
  ping -n 4 127.0.0.1 >nul
  exit /b
)

schtasks /create /tn "HypeVision" /tr "\"%~dp0CALISTIR.bat\"" /sc onlogon /rl limited /f >nul
if errorlevel 1 (
  echo Gorev olusturulamadi. Bu dosyayi sag tik ^> Yonetici olarak calistir ile deneyin.
  pause
  exit /b
)

powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Hype Vision.lnk');$s.TargetPath='%~dp0CALISTIR.bat';$s.WorkingDirectory='%~dp0';$s.IconLocation='%SystemRoot%\System32\shell32.dll,22';$s.WindowStyle=7;$s.Save()"

echo Tamam:
echo  - Windows'a her giriste Hype Vision otomatik baslayacak
echo  - Masaustune "Hype Vision" kisayolu eklendi
echo Kaldirmak icin:  OTOMATIK_BASLAT.bat kaldir
ping -n 6 127.0.0.1 >nul
