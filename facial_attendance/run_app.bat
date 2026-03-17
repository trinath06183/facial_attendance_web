@echo off
title SmartAttend Server Controller

echo [1/2] Starting Ngrok tunnel on port 8000...
:: Starts Ngrok in a separate, titled window so you can see the public URL
start "Ngrok_Tunnel" ngrok http 8000

echo [2/2] Starting Django Development Server...
echo ========================================================
echo [TIP] Press Ctrl+C in this window to stop the server.
echo       Ngrok will automatically close right after.
echo ========================================================

:: Runs Django directly in this main window. 
:: The script will "freeze" on this line as long as the server is running.
py manage.py runserver 0.0.0.0:8000

:: The commands below will ONLY run AFTER you stop the Django server
echo.
echo [CLEANUP] Django server stopped. Closing Ngrok...
taskkill /FI "WINDOWTITLE eq Ngrok_Tunnel*" /F /T > nul 2>&1

echo All processes successfully closed.
timeout /t 2 > nul