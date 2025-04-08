import streamlit as st
import os
import numpy as np
import time
import sys
import cv2
from PIL import Image

# Add parent directory to path for components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components directly using relative paths
from components.enrollment import EnrollmentComponent
from components.verification import VerificationComponent
from components.testing import TestingComponent
from models.face_model import FaceRecognitionModel

# Set page config
st.set_page_config(
    page_title="Facial Recognition with Emotion-Based Liveness Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_gpu():
    """Check if CUDA is available for GPU acceleration"""
    # Check if OpenCV was built with CUDA support
    cv_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
    
    # Check system for NVIDIA GPUs
    has_gpu = False
    gpu_name = "Unknown"
    
    try:
        # Try to detect NVIDIA GPU using Python subprocess
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            has_gpu = True
            gpu_name = result.stdout.strip()
    except:
        pass
    
    return has_gpu, cv_gpu, gpu_name

@st.cache_resource
def load_model():
    """Load and cache the face recognition model"""
    # Create data directories if they don't exist
    os.makedirs("app/data/embeddings", exist_ok=True)
    os.makedirs("app/data/temp", exist_ok=True)
    
    try:
        # Get GPU info
        has_gpu, cv_gpu, _ = check_gpu()
        
        # Create model with optimal settings based on hardware
        if cv_gpu:
            # If OpenCV has CUDA support, tell the model to use GPU
            model = FaceRecognitionModel(use_gpu=True)
        else:
            model = FaceRecognitionModel()
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    """Main application entry point"""
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">Facial Recognition with Emotion-Based Liveness Detection</div>', unsafe_allow_html=True)
    st.markdown("A proof-of-concept system for identity verification with emotion-based liveness checks")
    
    # Load the model
    with st.spinner("Loading facial recognition model..."):
        model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please restart the application.")
        return
    
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        
        # Mode selection
        app_mode = st.radio(
            "Select Mode",
            ["Enrollment", "Verification", "Testing & Analysis"]
        )
        
        # System info
        st.subheader("System Information")
        
        # Check GPU availability
        has_gpu, cv_gpu, gpu_name = check_gpu()
        
        if has_gpu:
            if cv_gpu:
                st.success("✅ GPU Acceleration Active")
                st.info(f"GPU: {gpu_name}")
            else:
                st.warning("⚠️ GPU detected but not used by OpenCV")
                st.info(f"GPU: {gpu_name}")
                st.info("OpenCV was not built with CUDA support")
        else:
            st.warning("⚠️ Running on CPU")
            st.info("No GPU detected or drivers not installed")
        
        # Display enrolled users
        st.subheader("Enrolled Users")
        embedded_users = _get_enrolled_users()
        if embedded_users:
            for user in embedded_users:
                st.write(f"• {user}")
        else:
            st.write("No users enrolled yet")
    
    # Main app content based on selected mode
    if app_mode == "Enrollment":
        enrollment_component = EnrollmentComponent(model)
        enrollment_component.render()
    
    elif app_mode == "Verification":
        verification_component = VerificationComponent(model)
        verification_component.render()
    
    else:  # Testing & Analysis
        testing_component = TestingComponent(model)
        testing_component.render()
    
    # Footer
    st.markdown('<div class="footer">Facial Recognition and Embedding Analysis PoC</div>', unsafe_allow_html=True)

def _get_enrolled_users():
    """Get list of enrolled users"""
    users = []
    embeddings_dir = "app/data/embeddings"
    
    if not os.path.exists(embeddings_dir):
        return users
    
    for filename in os.listdir(embeddings_dir):
        if filename.endswith(".pkl"):
            user_id = filename.split(".")[0]
            users.append(user_id)
    
    return users

# Run the main app when this script is executed directly
if __name__ == "__main__":
    main() 