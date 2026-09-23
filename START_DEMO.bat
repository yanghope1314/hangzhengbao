@echo off
setlocal enabledelayedexpansion
title Gong-e Hang-Jian  ::  Demo Launcher

REM ============================================================
REM  Gong-e Hang-Jian  ::  Demo Launcher
REM
REM  NOTE: This file is intentionally ASCII-only.
REM  Windows CMD reads .bat files in the system ANSI codepage
REM  (GBK on Chinese Windows); a UTF-8 .bat with Chinese text
REM  gets mangled and the commands break. ASCII avoids that.
REM
REM  NOTE: 'cd D:\path' only changes the current dir ON that
REM  drive -- it does NOT switch drives. Must use 'cd /d'.
REM
REM  NOTE: Auto-picks a free port. If a previous demo is still
REM  running, the port is busy -- this scans upward instead of
REM  failing with "Port is not available".
REM ============================================================

cd /d "%~dp0"

echo.
echo ============================================================
echo   Gong-e Hang-Jian  -  Trade Background Verification Demo
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv not found.
    echo         Make sure this file sits in the project root.
    echo.
    pause
    exit /b 1
)

if not exist "app\streamlit_app.py" (
    echo [ERROR] app\streamlit_app.py not found.
    echo.
    pause
    exit /b 1
)

REM ---- scan for a free port (8511, 8512, ... 8530) ----
set PORT=8511
:findport
netstat -ano | findstr LISTENING | findstr ":%PORT% " >nul 2>&1
if not errorlevel 1 (
    set /a PORT+=1
    if !PORT! gtr 8530 (
        echo [ERROR] No free port in 8511-8530.
        echo         Close some programs and retry.
        echo.
        pause
        exit /b 1
    )
    goto findport
)

echo   Project : %CD%
echo   Port    : %PORT%
echo   URL     : http://localhost:%PORT%
echo.
echo   To STOP the demo: press Ctrl + C in this window.
echo ============================================================
echo.

.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py --server.port %PORT% --browser.gatherUsageStats false

echo.
echo Demo stopped.
pause
