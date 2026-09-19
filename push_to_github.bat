@echo off
title Push Car Price Prediction to GitHub
cls
echo ========================================================
echo        PUSH CAR PRICE PREDICTION TO GITHUB
echo ========================================================
echo.
echo Default repository push link:
echo https://github.com/NaveenaMS/car-price-prediction.git
echo.
set /p repo_url="Press ENTER to use default link, or paste another GitHub repository link: "

if "%repo_url%"=="" (
    set repo_url=https://github.com/NaveenaMS/car-price-prediction.git
)

echo.
echo [*] Setting remote origin to: %repo_url%
git remote remove origin 2>nul
git remote add origin %repo_url%
git branch -M main

echo [*] Pushing project files to GitHub...
git push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo [SUCCESS] Project successfully pushed to GitHub!
) else (
    echo.
    echo [NOTE] If push failed:
    echo 1. Make sure you created the repository on GitHub first: https://github.com/new?name=car-price-prediction
    echo 2. Make sure you are signed in to your GitHub account.
)

echo.
pause
