import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
import os
from typing import Tuple, List, Dict, Optional

class EnrollmentComponent:
    def __init__(self, face_model):
        """
        Initialize the enrollment component
        
        Args:
            face_model: Face recognition model
        """
        self.face_model = face_model
        
        # Initialize from session state if available
        if "enrollment_images" in st.session_state:
            self.enrollment_images = st.session_state.enrollment_images
        else:
            self.enrollment_images = []
            
        if "enrollment_embeddings" in st.session_state:
            self.enrollment_embeddings = st.session_state.enrollment_embeddings
        else:
            self.enrollment_embeddings = []
            
        self.temp_image_path = "app/data/temp"
        os.makedirs(self.temp_image_path, exist_ok=True)
        
        # Define required images
        self.required_images = 5
    
    def render(self):
        """Render the enrollment UI component"""
        st.subheader("Enrollment")
        
        # User ID input
        user_id = st.text_input("User ID", key="enrollment_user_id")
        
        # Method selection
        enrollment_method = st.radio(
            "Enrollment Method",
            options=["Webcam", "Upload Images"],
            key="enrollment_method"
        )
        
        # Display enrollment progress
        st.write(f"Enrollment Progress: {len(self.enrollment_images)}/{self.required_images} images")
        self._display_enrollment_preview()
        
        if enrollment_method == "Webcam":
            self._webcam_enrollment()
        else:
            self._upload_enrollment()
        
        # Enrollment button
        if st.button("Complete Enrollment", key="complete_enrollment") and user_id:
            if len(self.enrollment_embeddings) < 3:
                st.error("Please capture at least 3 face images for enrollment")
                return
            
            try:
                self.face_model.enroll_user(user_id, self.enrollment_embeddings)
                st.success(f"Successfully enrolled user: {user_id}")
                
                # Reset enrollment data
                self.enrollment_images = []
                self.enrollment_embeddings = []
                
                # Clear session state
                if "enrollment_images" in st.session_state:
                    del st.session_state.enrollment_images
                if "enrollment_embeddings" in st.session_state:
                    del st.session_state.enrollment_embeddings
                
                # Force page refresh
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"Error enrolling user: {str(e)}")
    
    def _webcam_enrollment(self):
        """Handle webcam-based enrollment"""
        # Display webcam feed
        img_file_buffer = st.camera_input("Capture your face", key="enrollment_camera")
        
        capture_col, info_col = st.columns(2)
        
        with capture_col:  
            if img_file_buffer is not None:
                # Read image from buffer
                captured_img = Image.open(img_file_buffer)
                img_array = np.array(captured_img)
                
                # Check if this is a new image by comparing with last image
                is_new_image = True
                if self.enrollment_images and len(self.enrollment_images) > 0:
                    # Simple comparison - check if shapes are the same and pixels are similar
                    if img_array.shape == self.enrollment_images[-1].shape:
                        diff = np.mean(np.abs(img_array - self.enrollment_images[-1]))
                        if diff < 10:  # Small difference threshold
                            is_new_image = False
                
                # Process captured image if it's new
                if is_new_image and self._process_enrollment_image(img_array):
                    st.success(f"Face captured! ({len(self.enrollment_embeddings)}/{self.required_images})")
                    
                    # If we have enough images, suggest completion
                    if len(self.enrollment_embeddings) >= self.required_images:
                        st.info("You have captured enough images. Click 'Complete Enrollment' to finish.")
                else:
                    if not is_new_image:
                        st.warning("Please capture a different pose or expression")
                    else:
                        st.error("No face detected or error processing image")
        
        with info_col:
            if len(self.enrollment_images) < self.required_images:
                remaining = self.required_images - len(self.enrollment_images)
                st.info(f"Please capture {remaining} more image(s) with different poses/expressions.")
                st.markdown("""
                **Tips for better enrollment:**
                - Look straight at the camera
                - Try different angles (slight left/right)
                - Make sure your face is well-lit
                - Avoid extreme facial expressions for enrollment
                """)
    
    def _upload_enrollment(self):
        """Handle upload-based enrollment"""
        uploaded_files = st.file_uploader(
            "Upload face images", 
            accept_multiple_files=True,
            type=["jpg", "jpeg", "png"],
            key="enrollment_upload"
        )
        
        if uploaded_files:
            # Process only new files
            for uploaded_file in uploaded_files:
                # Read image
                img = Image.open(uploaded_file)
                img_array = np.array(img)
                
                # Process uploaded image if we need more images
                if len(self.enrollment_embeddings) < self.required_images:
                    if self._process_enrollment_image(img_array):
                        st.success(f"Face detected in {uploaded_file.name}")
                    else:
                        st.error(f"No face detected in {uploaded_file.name}")
                else:
                    st.info("Maximum number of enrollment images reached")
                    break
    
    def _process_enrollment_image(self, img_array: np.ndarray) -> bool:
        """
        Process an enrollment image
        
        Args:
            img_array: Image array
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Detect face and get embedding
            detected, face_info, embedding = self.face_model.detect_and_analyze(img_array)
            
            if not detected or embedding is None:
                return False
            
            # Store the image and embedding
            self.enrollment_images.append(img_array)
            self.enrollment_embeddings.append(embedding)
            
            # Save to session state to persist across reruns
            st.session_state.enrollment_images = self.enrollment_images
            st.session_state.enrollment_embeddings = self.enrollment_embeddings
            
            return True
        except Exception as e:
            st.error(f"Error processing enrollment image: {str(e)}")
            return False
    
    def _display_enrollment_preview(self):
        """Display preview of enrolled faces"""
        # Display preview
        if self.enrollment_images:
            # Display thumbnails
            cols = st.columns(min(len(self.enrollment_images), 5))
            for idx, (col, img) in enumerate(zip(cols, self.enrollment_images[-5:])):
                with col:
                    st.image(img, caption=f"Face {idx+1}", width=100)
        else:
            st.write("No face images captured yet")
    
    def reset(self):
        """Reset enrollment data"""
        self.enrollment_images = []
        self.enrollment_embeddings = []
        
        # Clear session state
        if "enrollment_images" in st.session_state:
            del st.session_state.enrollment_images
        if "enrollment_embeddings" in st.session_state:
            del st.session_state.enrollment_embeddings 