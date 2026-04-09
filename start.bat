@echo off
title AstroHoops Launcher
echo ============================================
echo   AstroHoops Analytics Platform
echo ============================================
echo.

:: Start FastAPI backend
echo [1/2] Starting FastAPI backend on http://127.0.0.1:8000 ...
cd /d "%~dp0"
start "AstroHoops Backend" cmd /k "cd /d "%~dp0backend" && ..\\.venv\\Scripts\\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload"

:: Give the backend a moment to boot
timeout /t 3 /nobreak > nul

:: Start Next.js frontend
echo [2/2] Starting Next.js frontend on http://localhost:3000 ...
start "AstroHoops Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo ============================================
echo   Both servers starting!
echo   Backend:  http://127.0.0.1:8000
echo   Frontend: http://localhost:3000
echo ============================================
echo.
echo You can close this window. The servers run
echo in their own terminal windows.
pause
