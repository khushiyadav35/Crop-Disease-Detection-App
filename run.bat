@echo off
echo ========================================================
echo  Kisan Drishti - Crop Disease Detection App for Farmers
echo  Author: Khushi Yadav (B.Tech CSE)
echo ========================================================
echo.
echo [*] Checking Python environment...
python --version
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

echo.
echo [*] Launching Kisan Drishti Web Application on http://localhost:8000 ...
echo [*] Starting FastAPI production server with Uvicorn...
echo.

start "" "http://localhost:8000"
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
pause
