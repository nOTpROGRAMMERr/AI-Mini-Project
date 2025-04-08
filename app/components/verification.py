import streamlit as st
import cv2
import numpy as np
import time
import random
from PIL import Image
import os
from typing import Tuple, List, Dict, Optional

class VerificationComponent:
    def __init__(self, face_model):
        """
        Initialize the verification component
        
        Args:
            face_model: Face recognition model
        """
        self.face_model = face_model
        self.emotions = ["neutral", "happy", "sad", "surprised", "angry"]
        self.emotion_prompts = {
            "angry": "Show an angry expression",
            "happy": "Smile for the camera",
            "sad": "Show a sad expression",
            "surprised": "Show a surprised expression",
            "neutral": "Show a neutral expression"
        }
        self.emotion_instructions = {
            "angry": "Furrow your eyebrows and frown with your mouth.",
            "happy": "Show your teeth in a genuine smile and slightly squint your eyes.",
            "sad": "Turn down the corners of your mouth and look downward.",
            "surprised": "Raise your eyebrows and open your mouth slightly.",
            "neutral": "Relax your facial muscles with a normal resting expression."
        }
    
    def render(self):
        """Render the verification UI component"""
        st.subheader("Verification")
        
        # User ID input
        user_id = st.text_input("User ID", key="verification_user_id")
        
        # Get or generate the target emotion
        if "target_emotion" not in st.session_state:
            st.session_state.target_emotion = random.choice(self.emotions)
        
        target_emotion = st.session_state.target_emotion
        
        # Show the emotion prompt
        st.markdown(f"### Liveness Check: {self.emotion_prompts[target_emotion]}")
        st.info(f"**How to show '{target_emotion}'**: {self.emotion_instructions[target_emotion]}")
        
        # Method selection
        verification_method = st.radio(
            "Verification Method",
            options=["Webcam", "Upload Image"],
            key="verification_method"
        )
        
        if verification_method == "Webcam":
            self._webcam_verification(user_id, target_emotion)
        else:
            self._upload_verification(user_id, target_emotion)
        
        # New verification button
        if st.button("New Verification", key="new_verification"):
            # Generate a new target emotion
            st.session_state.target_emotion = random.choice(self.emotions)
            st.experimental_rerun()
    
    def _webcam_verification(self, user_id: str, target_emotion: str):
        """
        Handle webcam-based verification
        
        Args:
            user_id: User ID to verify
            target_emotion: Target emotion for liveness check
        """
        if not user_id:
            st.warning("Please enter a User ID to verify")
            return
        
        # Display webcam feed
        img_file_buffer = st.camera_input("Verify your face", key="verification_camera")
        
        if img_file_buffer is not None and st.button("Verify", key="verify_webcam"):
            # Read image from buffer
            captured_img = Image.open(img_file_buffer)
            img_array = np.array(captured_img)
            
            # Process verification
            self._process_verification(img_array, user_id, target_emotion)
    
    def _upload_verification(self, user_id: str, target_emotion: str):
        """
        Handle upload-based verification
        
        Args:
            user_id: User ID to verify
            target_emotion: Target emotion for liveness check
        """
        if not user_id:
            st.warning("Please enter a User ID to verify")
            return
        
        uploaded_file = st.file_uploader(
            "Upload face image", 
            type=["jpg", "jpeg", "png"],
            key="verification_upload"
        )
        
        if uploaded_file is not None and st.button("Verify", key="verify_upload"):
            # Read image
            img = Image.open(uploaded_file)
            img_array = np.array(img)
            
            # Process verification
            self._process_verification(img_array, user_id, target_emotion)
    
    def _process_verification(self, img_array: np.ndarray, user_id: str, target_emotion: str):
        """
        Process a verification image
        
        Args:
            img_array: Image array
            user_id: User ID to verify
            target_emotion: Target emotion for liveness check
        """
        try:
            # Detect face and get embedding
            detected, face_info, embedding = self.face_model.detect_and_analyze(img_array)
            
            if not detected or embedding is None:
                st.error("No face detected in the image")
                return
            
            # Verify identity
            identity_match, similarity = self.face_model.verify_identity(embedding, user_id)
            
            # Verify emotion (liveness check)
            emotion_match, emotions = self.face_model.verify_emotion(img_array, target_emotion)
            
            # Display results
            self._display_verification_results(
                img_array, 
                identity_match, 
                similarity, 
                emotion_match, 
                emotions, 
                target_emotion
            )
            
        except Exception as e:
            st.error(f"Error processing verification: {str(e)}")
    
    def _display_verification_results(
        self, 
        img_array: np.ndarray, 
        identity_match: bool, 
        similarity: float, 
        emotion_match: bool, 
        emotions: Dict[str, float], 
        target_emotion: str
    ):
        """
        Display verification results
        
        Args:
            img_array: Image array
            identity_match: Whether identity matched
            similarity: Identity similarity score
            emotion_match: Whether emotion matched
            emotions: Emotion scores
            target_emotion: Target emotion
        """
        # Create columns for results display
        img_col, results_col = st.columns([1, 1])
        
        with img_col:
            st.image(img_array, caption="Verification Image", use_column_width=True)
        
        with results_col:
            # Overall verification result
            verification_success = identity_match and emotion_match
            
            if verification_success:
                st.success("✅ Verification Successful")
            else:
                st.error("❌ Verification Failed")
            
            # Identity verification details
            st.subheader("Identity Verification")
            if identity_match:
                st.markdown(f"**Result:** ✅ Match")
            else:
                st.markdown(f"**Result:** ❌ No Match")
            st.markdown(f"**Similarity Score:** {similarity:.2f}")
            
            # Emotion verification details
            st.subheader("Liveness Check (Emotion)")
            if emotion_match:
                st.markdown(f"**Result:** ✅ Emotion Match")
            else:
                st.markdown(f"**Result:** ❌ Emotion Mismatch")
            
            st.markdown(f"**Target Emotion:** {target_emotion}")
            
            # Display emotion scores as a bar chart
            st.markdown("**Detected Emotions:**")
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            for emotion, score in sorted_emotions:
                st.progress(float(score))
                st.write(f"{emotion}: {score:.2f}")
    
    def reset(self):
        """Reset verification state"""
        if "target_emotion" in st.session_state:
            del st.session_state.target_emotion 