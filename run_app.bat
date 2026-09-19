@echo off
title Car Price Prediction System
cls
echo ========================================================
echo        CAR PRICE PREDICTION SYSTEM LAUNCHER
echo ========================================================
echo.
echo [1] Launch Full-Stack Web Application (Backend + Frontend)
echo [2] Launch Streamlit Interactive Application
echo [3] Retrain ML Models & Regenerate Plots
echo [4] Exit
echo.
set /p choice="Choose an option (1-4): "

if "%choice%"=="1" (
    echo.
    echo [*] Starting Flask REST API Backend and serving Web UI...
    echo [*] Open your browser at http://localhost:5000
    python backend/server.py
    pause
) else if "%choice%"=="2" (
    echo.
    echo [*] Starting Streamlit App...
    streamlit run app.py
    pause
) else if "%choice%"=="3" (
    echo.
    echo [*] Retraining models and evaluating...
    python src/train_model.py
    pause
) else (
    exit
)
