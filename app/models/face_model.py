import os
import numpy as np
import cv2
import sys
import pickle
from typing import Dict, List, Tuple, Optional, Union

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utility functions
from utils.face_utils import (
    load_embeddings, 
    save_embeddings, 
    compare_embeddings, 
    analyze_emotion,
    detect_face,
    get_face_embedding
)

class FaceRecognitionModel:
    """
    Wrapper model that handles both face recognition and emotion analysis
    """
    def __init__(self, 
                recognition_threshold: float = 0.6,
                emotion_threshold: float = 0.4,
                use_gpu: bool = False):
        """
        Initialize the face recognition model
        
        Args:
            recognition_threshold: Threshold for face recognition
            emotion_threshold: Threshold for emotion validation
            use_gpu: Whether to use GPU acceleration if available
        """
        self.recognition_threshold = recognition_threshold
        self.emotion_threshold = emotion_threshold
        self.use_gpu = use_gpu
        
        # Check if GPU acceleration is available via OpenCV
        self.has_gpu = False
        if use_gpu:
            self.has_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
            if self.has_gpu:
                print("GPU acceleration is enabled for OpenCV")
            else:
                print("GPU acceleration requested but not available")
        
        # Emotions mapping - using simplified approach
        self.emotion_map = {
            "neutral": "neutral",
            "happy": "happy",
            "sad": "sad",
            "surprised": "surprised",
            "angry": "angry"
        }
        
        # Load user embeddings if available
        self.user_db = {}
        self._load_user_db()
    
    def _load_user_db(self, directory: str = "app/data/embeddings"):
        """
        Load all user embeddings into memory
        
        Args:
            directory: Directory containing user embeddings
        """
        try:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                return
                
            for filename in os.listdir(directory):
                if filename.endswith(".pkl"):
                    user_id = filename.split(".")[0]
                    embeddings = load_embeddings(user_id, directory)
                    self.user_db[user_id] = embeddings
        except Exception as e:
            print(f"Error loading user database: {str(e)}")
    
    def detect_and_analyze(self, image: np.ndarray) -> Tuple[bool, Optional[dict], Optional[np.ndarray]]:
        """
        Detect and analyze faces in an image
        
        Args:
            image: Input image
            
        Returns:
            Tuple containing:
            - detected: Whether a face was detected
            - face_info: Information about the detected face
            - face_embedding: Embedding of the detected face
        """
        try:
            # Ensure image is RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # Apply GPU preprocessing if available
            if self.has_gpu:
                # Upload to GPU memory
                gpu_image = cv2.cuda_GpuMat()
                gpu_image.upload(image)
                
                # Apply GPU-accelerated preprocessing (e.g., resize, normalize)
                # If needed for your specific model
                
                # Download back to CPU for detection (unless detect_face supports GPU)
                image = gpu_image.download()
            
            # Detect face
            detected, face_img, face_obj = detect_face(image, use_gpu=self.has_gpu)
            
            if not detected or face_img is None:
                return False, None, None
            
            # Get embedding
            embedding = get_face_embedding(face_img, use_gpu=self.has_gpu)
            
            # Create face info dictionary
            if face_obj and "facial_area" in face_obj:
                x, y, w, h = face_obj["facial_area"]
                face_info = {
                    "bbox": [x, y, x+w, y+h],
                    "confidence": face_obj.get("confidence", 0.0),
                }
            else:
                face_info = {
                    "bbox": [0, 0, face_img.shape[1], face_img.shape[0]],
                    "confidence": 1.0,
                }
            
            return True, face_info, embedding
        except Exception as e:
            print(f"Face detection error: {str(e)}")
            return False, None, None
    
    def verify_identity(self, embedding: np.ndarray, user_id: str) -> Tuple[bool, float]:
        """
        Verify if the embedding matches the stored embeddings for the user
        
        Args:
            embedding: Face embedding to verify
            user_id: ID of the user to compare against
            
        Returns:
            Tuple containing:
            - match: Whether the identity matches
            - similarity: Similarity score
        """
        # Load embeddings if not in memory
        if user_id not in self.user_db:
            self.user_db[user_id] = load_embeddings(user_id)
        
        # Compare embeddings
        return compare_embeddings(
            source_embedding=embedding,
            target_embeddings=self.user_db.get(user_id, []),
            threshold=self.recognition_threshold
        )
    
    def predict_emotion(self, image: np.ndarray) -> Dict[str, float]:
        """
        Predict emotions in the face image
        
        Args:
            image: Face image
            
        Returns:
            Dictionary mapping emotion names to confidence scores
        """
        return analyze_emotion(image, use_gpu=self.has_gpu)
    
    def verify_emotion(self, image: np.ndarray, target_emotion: str) -> Tuple[bool, Dict[str, float]]:
        """
        Verify if the face in the image shows the target emotion
        
        Args:
            image: Face image
            target_emotion: Expected emotion
            
        Returns:
            Tuple containing:
            - match: Whether the emotion matches
            - emotions: Dictionary of emotion scores
        """
        emotions = self.predict_emotion(image)
        observed_emotion, score = max(emotions.items(), key=lambda x: x[1])
        
        match = (observed_emotion == target_emotion and score >= self.emotion_threshold)
        
        return match, emotions
    
    def enroll_user(self, user_id: str, face_embeddings: List[np.ndarray], directory: str = "app/data/embeddings"):
        """
        Enroll a user by saving their face embeddings
        
        Args:
            user_id: User ID
            face_embeddings: List of face embeddings
            directory: Directory to save embeddings
        """
        # Save embeddings to disk
        save_embeddings(user_id, face_embeddings, directory)
        
        # Update in-memory database
        self.user_db[user_id] = face_embeddings 