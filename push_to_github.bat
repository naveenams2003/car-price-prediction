@echo off
title Push Car Price Prediction to GitHub
cls
echo ========================================================
echo        PUSH CAR PRICE PREDICTION TO GITHUB
echo ========================================================
echo.
echo [*] Pushing latest changes to https://github.com/naveenams2003/car-price-prediction.git ...
git add .
git commit -m "Update project files" 2>nul
git push origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo [SUCCESS] Project successfully updated on GitHub!
) else (
    echo.
    echo [ERROR] Push failed. Please check your internet connection or GitHub authentication.
)

echo.
pause
