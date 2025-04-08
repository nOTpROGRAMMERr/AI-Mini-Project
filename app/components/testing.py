import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from typing import Tuple, List, Dict, Optional

class TestingComponent:
    def __init__(self, face_model):
        """
        Initialize the testing component
        
        Args:
            face_model: Face recognition model
        """
        self.face_model = face_model
    
    def render(self):
        """Render the testing UI component"""
        st.subheader("Testing & Analysis")
        
        # Method selection
        test_method = st.radio(
            "Testing Method",
            options=["Webcam", "Upload Image"],
            key="test_method"
        )
        
        if test_method == "Webcam":
            self._webcam_testing()
        else:
            self._upload_testing()
    
    def _webcam_testing(self):
        """Handle webcam-based testing"""
        # Display webcam feed
        img_file_buffer = st.camera_input("Test with your face", key="test_camera")
        
        if img_file_buffer is not None and st.button("Analyze", key="analyze_webcam"):
            # Read image from buffer
            captured_img = Image.open(img_file_buffer)
            img_array = np.array(captured_img)
            
            # Process testing
            self._process_testing(img_array)
    
    def _upload_testing(self):
        """Handle upload-based testing"""
        uploaded_file = st.file_uploader(
            "Upload face image", 
            type=["jpg", "jpeg", "png"],
            key="test_upload"
        )
        
        if uploaded_file is not None and st.button("Analyze", key="analyze_upload"):
            # Read image
            img = Image.open(uploaded_file)
            img_array = np.array(img)
            
            # Process testing
            self._process_testing(img_array)
    
    def _process_testing(self, img_array: np.ndarray):
        """
        Process a testing image
        
        Args:
            img_array: Image array
        """
        try:
            # Detect face and get embedding
            detected, face_info, embedding = self.face_model.detect_and_analyze(img_array)
            
            if not detected or embedding is None:
                st.error("No face detected in the image")
                return
            
            # Get emotion analysis
            emotions = self.face_model.predict_emotion(img_array)
            
            # Display results
            self._display_testing_results(img_array, face_info, embedding, emotions)
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    def _display_testing_results(
        self, 
        img_array: np.ndarray, 
        face_info: Dict, 
        embedding: np.ndarray, 
        emotions: Dict[str, float]
    ):
        """
        Display testing results
        
        Args:
            img_array: Image array
            face_info: Face detection information
            embedding: Face embedding
            emotions: Emotion scores
        """
        # Create columns for results display
        img_col, results_col = st.columns(2)
        
        with img_col:
            # Display image with face bounding box
            self._display_face_with_box(img_array, face_info)
        
        with results_col:
            # Display emotion results
            st.subheader("Emotion Analysis")
            
            # Create bar chart of emotions
            self._plot_emotions_bar(emotions)
            
            # Dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            st.markdown(f"**Dominant Emotion:** {dominant_emotion[0]} ({dominant_emotion[1]:.2f})")
            
            # Display embedding stats
            st.subheader("Embedding Analysis")
            st.markdown(f"**Embedding Dimension:** {embedding.shape[0]}")
            
            # Embedding visualization (dimensionality reduction not included here)
            self._plot_embedding_stats(embedding)
    
    def _display_face_with_box(self, img_array: np.ndarray, face_info: Dict):
        """
        Display image with face bounding box
        
        Args:
            img_array: Image array
            face_info: Face detection information
        """
        # Create a copy of the image
        display_img = img_array.copy()
        
        if face_info and "bbox" in face_info:
            # Get bounding box coordinates
            bbox = face_info["bbox"]
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Draw bounding box
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display the image
        st.image(display_img, caption="Detected Face", use_column_width=True)
    
    def _plot_emotions_bar(self, emotions: Dict[str, float]):
        """
        Plot emotion scores as a bar chart
        
        Args:
            emotions: Emotion scores
        """
        # Sort emotions by score
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        # Display emotion scores as a bar chart
        for emotion, score in sorted_emotions:
            st.progress(float(score))
            st.write(f"{emotion}: {score:.2f}")
    
    def _plot_embedding_stats(self, embedding: np.ndarray):
        """
        Plot embedding statistics
        
        Args:
            embedding: Face embedding
        """
        # Create a figure for embedding visualization
        fig, ax = plt.subplots(figsize=(8, 2))
        
        # Plot embedding distribution (simplified)
        ax.hist(embedding, bins=20, alpha=0.7)
        ax.set_title("Embedding Value Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        
        # Display the plot
        st.pyplot(fig)
        
        # Show embedding stats
        st.markdown(f"**Mean:** {np.mean(embedding):.4f}")
        st.markdown(f"**Std Dev:** {np.std(embedding):.4f}")
        st.markdown(f"**Min:** {np.min(embedding):.4f}")
        st.markdown(f"**Max:** {np.max(embedding):.4f}") 