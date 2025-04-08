# Getting Started

This document provides step-by-step instructions to set up and run the Facial Recognition and Embedding Analysis PoC.

## Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended)
- Webcam (for live testing)

## Installation

### Windows

1. Clone or download this repository
2. Double-click `run_app.bat` to:
   - Create a virtual environment
   - Install dependencies
   - Start the application

### macOS/Linux

1. Clone or download this repository
2. Open a terminal in the project directory
3. Make the script executable (if needed):
   ```
   chmod +x run_app.sh
   ```
4. Run the script:
   ```
   ./run_app.sh
   ```

### Manual Setup

If you prefer to set up manually:

1. Create a virtual environment:
   ```
   python -m venv venv
   ```
2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the application:
   ```
   streamlit run app/app.py
   ```

## Using the Application

Once started, the application should be accessible in your browser at:
```
http://localhost:8501
```

### Enrollment

1. Select the "Enrollment" mode
2. Enter a unique User ID
3. Use webcam or upload images to capture face data
4. Click "Complete Enrollment" when done

### Verification

1. Select the "Verification" mode
2. Enter the User ID to verify
3. Follow the displayed emotion prompt
4. Use webcam or upload image to verify both identity and emotion

### Testing & Analysis

Use this mode to test the facial recognition and emotion detection systems with arbitrary images.

## Troubleshooting

- If you encounter issues with model loading, ensure your CUDA drivers are up to date
- For webcam access issues, check your browser permissions
- If the application fails to start, try running it manually with the command `streamlit run app/app.py` 