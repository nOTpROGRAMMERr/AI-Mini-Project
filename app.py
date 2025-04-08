"""
Facial Recognition with Emotion-Based Liveness Detection
This is the main entry point for the application
"""
import os
import streamlit.web.cli as stcli
import sys

# Make sure paths exist
os.makedirs("app/data/embeddings", exist_ok=True)
os.makedirs("app/data/temp", exist_ok=True)

# Run the Streamlit app directly
if __name__ == "__main__":
    sys.argv = ["streamlit", "run", os.path.join("app", "app.py")]
    sys.exit(stcli.main()) 