"""
Main entry point for the Facial Recognition application
This script should be run directly with Python
"""
import os
import subprocess
import sys

# Make sure the data directories exist
os.makedirs("app/data/embeddings", exist_ok=True)
os.makedirs("app/data/temp", exist_ok=True)

def main():
    """Run the Streamlit application"""
    print("Starting Facial Recognition with Emotion-Based Liveness Detection...")
    print("Access the application at http://localhost:8501")
    
    # Run the Streamlit application directly as a subprocess
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "app/app.py",
        "--server.port=8501", 
        "--server.headless=false"
    ])

if __name__ == "__main__":
    main() 