@echo off
echo Starting Facial Recognition with Emotion-Based Liveness Detection...
echo.
echo This script will install required dependencies and start the application.
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.8 or higher.
    goto :eof
)

REM Check if venv exists, if not create it
if not exist venv\Scripts\activate.bat (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
mkdir app\data\embeddings 2>nul
mkdir app\data\temp 2>nul

REM Run the application
echo Starting the application...
echo.
echo Access the application at http://localhost:8501
python app.py

pause 