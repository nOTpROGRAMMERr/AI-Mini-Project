import os
import cv2
import numpy as np
import pickle
from typing import Dict, List, Tuple, Union, Optional

# Supported emotion list (using facial features)
EMOTIONS = ["neutral", "happy", "sad", "surprised", "angry"]

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(image: np.ndarray, use_gpu: bool = False) -> Tuple[bool, Optional[np.ndarray], Optional[dict]]:
    """
    Detects face in the image and returns the face region
    
    Args:
        image: Input image as numpy array
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Tuple containing:
        - detected: Boolean indicating if face was detected
        - face_img: Cropped face image if detected, else None
        - face_obj: Face object with detection details
    """
    try:
        # Convert to grayscale for face detection
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Use GPU acceleration if available
        if use_gpu:
            try:
                # Create a GPU version of the face detector if not already created
                gpu_cascade = cv2.cuda.CascadeClassifier_create(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                
                # Upload image to GPU
                gpu_gray = cv2.cuda_GpuMat()
                gpu_gray.upload(gray)
                
                # Detect faces on GPU
                gpu_faces = gpu_cascade.detectMultiScale(gpu_gray)
                
                # Download result to CPU
                faces = gpu_faces.download()
            except Exception as e:
                print(f"GPU face detection failed, falling back to CPU: {str(e)}")
                # Fall back to CPU
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
        else:
            # Use CPU for detection
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
        
        if len(faces) == 0:
            return False, None, None
        
        # Get the largest face
        if len(faces) > 1:
            # Find the largest face by area
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
        else:
            x, y, w, h = faces[0]
        
        # Extract face image
        face_img = image[y:y+h, x:x+w]
        
        # Create face object with metadata
        face_obj = {
            "facial_area": (x, y, w, h),
            "confidence": 1.0
        }
        
        return True, face_img, face_obj
    except Exception as e:
        print(f"Face detection error: {str(e)}")
        return False, None, None

def get_face_embedding(face_img: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """
    Extracts facial embedding (vector representation) from face image
    
    Args:
        face_img: Cropped face image
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        embedding: Facial embedding vector
    """
    # Convert to grayscale if needed
    if len(face_img.shape) > 2:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img
    
    # Use GPU if available
    if use_gpu:
        try:
            # Upload to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(gray)
            
            # Resize on GPU
            gpu_resized = cv2.cuda.resize(gpu_img, (100, 100))
            
            # Download back to CPU
            resized = gpu_resized.download()
        except Exception as e:
            print(f"GPU embedding extraction failed, falling back to CPU: {str(e)}")
            # Fall back to CPU
            resized = cv2.resize(gray, (100, 100))
    else:
        # Resize for consistent embedding size
        resized = cv2.resize(gray, (100, 100))
    
    # Flatten the image to create a simple embedding
    # For a real app, you'd use a proper face embedding model
    embedding = resized.flatten().astype(np.float32)
    
    # Normalize the embedding
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding

def analyze_emotion(face_img: np.ndarray, use_gpu: bool = False) -> Dict[str, float]:
    """
    Analyzes the emotion in a face image
    
    Args:
        face_img: Cropped face image
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        emotions: Dictionary of emotion probabilities
    """
    # Convert to grayscale if needed
    if len(face_img.shape) > 2:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img
    
    # Use GPU if available for preprocessing
    if use_gpu:
        try:
            # Upload to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(gray)
            
            # Apply GPU-accelerated preprocessing
            gpu_img = cv2.cuda.resize(gpu_img, (100, 100))
            
            # Download back to CPU
            gray = gpu_img.download()
            
            # GPU edge detection
            gpu_edges = cv2.cuda_GpuMat()
            gpu_edges.upload(gray)
            gpu_edges = cv2.cuda.createCannyEdgeDetector(100, 200).detect(gpu_edges)
            edges = gpu_edges.download()
        except Exception as e:
            print(f"GPU emotion analysis failed, falling back to CPU: {str(e)}")
            # Fall back to CPU
            gray = cv2.resize(gray, (100, 100))
            edges = cv2.Canny(gray, 100, 200)
    else:
        # Simple emotion detection using basic image features
        gray = cv2.resize(gray, (100, 100))
        edges = cv2.Canny(gray, 100, 200)
    
    # Initialize emotion scores
    emotions = {
        "neutral": 0.3,  # Reduced default for neutral
        "happy": 0.1,
        "sad": 0.1,
        "surprised": 0.1,
        "angry": 0.1
    }
    
    # Use edge detection as a very simple proxy for facial features
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    
    # Use image variance as a simple measure of expression intensity
    variance = np.var(gray)
    
    # Approximate emotion based on simple image features
    if edge_density > 0.12:
        # More edges could indicate surprised or angry
        emotions["surprised"] = 0.5
        emotions["angry"] = 0.3
    elif variance > 1500:  # Lowered threshold to detect happiness more easily
        # Higher variance might indicate happiness
        emotions["happy"] = 0.7
        emotions["neutral"] = 0.1
    else:
        # Lower variance might indicate neutral or sad
        emotions["neutral"] = 0.5
        emotions["sad"] = 0.3
    
    # Normalize emotions to sum to 1
    total = sum(emotions.values())
    emotions = {k: v/total for k, v in emotions.items()}
    
    return emotions

def save_embeddings(user_id: str, embeddings: List[np.ndarray], directory: str = "app/data/embeddings"):
    """
    Saves user embeddings to disk
    
    Args:
        user_id: Unique identifier for the user
        embeddings: List of embedding vectors
        directory: Directory to save the embeddings
    """
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{user_id}.pkl")
    
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(user_id: str, directory: str = "app/data/embeddings") -> List[np.ndarray]:
    """
    Loads user embeddings from disk
    
    Args:
        user_id: Unique identifier for the user
        directory: Directory where embeddings are stored
    
    Returns:
        embeddings: List of embedding vectors
    """
    file_path = os.path.join(directory, f"{user_id}.pkl")
    
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    
    return embeddings

def compare_embeddings(source_embedding: np.ndarray, target_embeddings: List[np.ndarray], 
                      threshold: float = 0.6, metric: str = "cosine") -> Tuple[bool, float]:
    """
    Compares a source embedding to a list of target embeddings
    
    Args:
        source_embedding: The embedding to compare
        target_embeddings: List of embeddings to compare against
        threshold: Similarity threshold for matching
        metric: Distance metric to use (cosine, euclidean, etc.)
    
    Returns:
        Tuple containing:
        - match: Boolean indicating if match was found
        - similarity: Highest similarity score
    """
    if not target_embeddings:
        return False, 0.0
    
    best_similarity = 0.0
    
    for target in target_embeddings:
        if metric == "cosine":
            # Calculate cosine similarity
            similarity = np.dot(source_embedding, target) / (
                np.linalg.norm(source_embedding) * np.linalg.norm(target)
            )
        elif metric == "euclidean":
            # Calculate Euclidean similarity (inversely related to distance)
            distance = np.linalg.norm(source_embedding - target)
            similarity = 1.0 / (1.0 + distance)
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")
        
        if similarity > best_similarity:
            best_similarity = similarity
    
    return best_similarity >= threshold, float(best_similarity)

def get_dominant_emotion(emotions: Dict[str, float]) -> Tuple[str, float]:
    """
    Gets the dominant emotion from an emotions dictionary
    
    Args:
        emotions: Dictionary of emotion probabilities
    
    Returns:
        Tuple containing:
        - emotion: Name of dominant emotion
        - score: Probability score of the dominant emotion
    """
    if not emotions:
        return "unknown", 0.0
    
    dominant_emotion = max(emotions.items(), key=lambda x: x[1])
    return dominant_emotion 