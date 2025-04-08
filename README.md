# Facial Recognition with Emotion Liveness Detection

A proof-of-concept system for identity verification with emotion-based liveness detection using facial analysis.

## Overview

This project combines facial recognition with emotion detection to:
1. Verify a user's identity through facial embeddings
2. Confirm liveness by requesting and validating specific facial expressions

## Features

- **Face enrollment**: Register your face for identity verification
- **Facial recognition**: Verify identity using facial embeddings
- **Emotion-based liveness detection**: System prompts for specific emotions (happy, sad, angry, surprised, neutral)
- **Detailed emotion guidance**: Clear instructions on how to express each emotion
- **Real-time webcam integration**: Use your webcam for instant verification
- **Upload option**: Alternatively upload photos for offline testing

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/nOTpROGRAMMERr/AI-Mini-Project.git
   cd AI-Mini-Project
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python run_app.py
   ```
   
   Or use the provided scripts:
   - Windows: `run_app.bat`
   - Linux/Mac: `run_app.sh`

## Usage Guide

1. **Enrollment**: First, enroll your face by:
   - Entering a User ID
   - Capturing multiple images of your face from different angles
   - Following the emotion prompts for multi-factor verification

2. **Verification**: Verify your identity by:
   - Entering your User ID
   - Following the specific emotion prompt shown on screen
   - Capturing your face with the requested expression

3. **Testing & Analysis**: Use the analysis mode to:
   - Test emotion detection accuracy
   - View detailed emotion scores
   - Analyze facial recognition confidence

## Project Structure

- `app/app.py`: Main application entry point
- `app/components/`: UI components
- `app/utils/`: Utility functions for face processing and emotion detection
- `app/models/`: Face recognition model wrappers
- `app/data/embeddings/`: Storage for enrolled facial embeddings

## Hardware Requirements

- Intel i5 processor or equivalent
- NVIDIA GPU with CUDA support (optional, improves performance)
- 8GB RAM minimum

## Recent Improvements

- Enhanced emotion detection algorithms
- Detailed guidance for expressing each emotion
- Improved happy expression detection
- Adjusted threshold parameters for better verification accuracy
- Added comprehensive .gitignore configuration 