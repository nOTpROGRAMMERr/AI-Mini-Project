# Facial Recognition and Embedding Analysis PoC

A proof-of-concept system for identity verification with emotion-based liveness detection using facial analysis.

## Overview

This project combines facial recognition with emotion detection to:
1. Verify a user's identity through facial embeddings
2. Confirm liveness by requesting and validating specific facial expressions

## Features

- Face enrollment for identity registration
- Facial recognition with embedding comparison
- Emotion detection (anger, fear, happiness, sadness, surprise, neutral)
- Real-time webcam integration
- Streamlit user interface for easy interaction

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```
streamlit run app/app.py
```

## Project Structure

- `app/app.py`: Main application entry point
- `app/components/`: UI components
- `app/utils/`: Utility functions for face processing
- `app/models/`: Model wrappers
- `app/data/`: Storage for enrolled faces and settings

## Hardware Requirements

- Intel i9 13th Gen processor or equivalent
- NVIDIA GPU with CUDA support (tested on RTX 3050)
- 16GB RAM minimum 