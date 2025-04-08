#!/bin/bash

echo "Starting Facial Recognition with Emotion-Based Liveness Detection..."
echo
echo "This script will install required dependencies and start the application."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p app/data/embeddings
mkdir -p app/data/temp

# Run the application
echo "Starting the application..."
echo
echo "Access the application at http://localhost:8501"
streamlit run app.py 