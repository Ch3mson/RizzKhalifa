"""
Face recognition core module.
Provides functionality for recognizing and comparing faces.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import time

class FaceRecognition:
    """
    Core functionality for face recognition.
    Includes methods for face matching, similarity calculation, and identity management.
    """
    
    def __init__(self, face_analyzer, recognition_threshold=0.5):
        """
        Initialize the face recognition module.
        
        Args:
            face_analyzer: An instance of ImprovedFaceAnalysis or similar
            recognition_threshold: Threshold for face matching (lower = more strict)
        """
        self.logger = logging.getLogger("facial_recognition.recognition")
        self.face_analyzer = face_analyzer
        self.recognition_threshold = recognition_threshold
        
        # Face tracking state
        self.current_face_name = None
        self.current_face_embedding = None
        self.last_face_check_time = 0
        self.face_recheck_interval = 60  # seconds
        
    def set_recheck_interval(self, seconds: int) -> None:
        """
        Set the interval for rechecking face identity.
        
        Args:
            seconds: Interval in seconds
        """
        self.face_recheck_interval = max(1, seconds)
        self.logger.info(f"Set face recheck interval to {self.face_recheck_interval} seconds")
    
    def get_recheck_interval(self) -> int:
        """
        Get the current recheck interval.
        
        Returns:
            int: Current recheck interval in seconds
        """
        return self.face_recheck_interval
    
    def should_recheck_face(self) -> bool:
        """
        Check if it's time to recheck the current face.
        
        Returns:
            bool: True if it's time to recheck, False otherwise
        """
        current_time = time.time()
        time_since_last_check = current_time - self.last_face_check_time
        
        return time_since_last_check >= self.face_recheck_interval
    
    def update_current_face(self, face_name: str, face_embedding: np.ndarray, face_image: np.ndarray = None) -> None:
        """
        Update the current face being tracked.
        
        Args:
            face_name: Name/ID of the face
            face_embedding: Face embedding vector
            face_image: Optional face image data
        """
        self.current_face_name = face_name
        self.current_face_embedding = face_embedding
        self.last_face_check_time = time.time()
        
        self.logger.info(f"Updated current face to {face_name}")
    
    def is_same_face(self, face_embedding: np.ndarray, threshold: float = None) -> bool:
        """
        Check if a face embedding matches the current face.
        
        Args:
            face_embedding: Face embedding to compare
            threshold: Optional similarity threshold (overrides default)
            
        Returns:
            bool: True if it's the same face, False otherwise
        """
        if self.current_face_embedding is None or face_embedding is None:
            return False
            
        if threshold is None:
            threshold = self.recognition_threshold
            
        similarity = self._calculate_similarity(self.current_face_embedding, face_embedding)
        
        return similarity > threshold
    
    def recognize_face(self, face_embeddings: Dict[str, np.ndarray], new_embedding: np.ndarray) -> Tuple[str, float]:
        """
        Find the best match for a face embedding in a collection of known faces.
        
        Args:
            face_embeddings: Dictionary of known face embeddings
            new_embedding: New face embedding to match
            
        Returns:
            Tuple[str, float]: Name of the best match and similarity score
        """
        best_match = None
        best_similarity = 0
        
        for name, data in face_embeddings.items():
            embedding = None
            
            # Handle different data formats
            if isinstance(data, dict) and 'embedding' in data:
                embedding = data['embedding']
            elif isinstance(data, np.ndarray):
                embedding = data
                
            if embedding is not None:
                similarity = self._calculate_similarity(embedding, new_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
        
        # Check if the best match exceeds the threshold
        if best_match and best_similarity > self.recognition_threshold:
            self.logger.info(f"Recognized face as {best_match} with similarity {best_similarity:.4f}")
            return best_match, best_similarity
        else:
            self.logger.info(f"No match found (best similarity: {best_similarity:.4f})")
            return None, best_similarity
    
    def detect_faces(self, frame: np.ndarray, max_faces: int = 0, min_face_size: int = 40) -> List:
        """
        Detect faces in an image.
        
        Args:
            frame: Input image
            max_faces: Maximum number of faces to detect (0 for unlimited)
            min_face_size: Minimum face size in pixels
            
        Returns:
            List: Detected faces with embeddings and landmarks
        """
        if frame is None:
            self.logger.error("Cannot detect faces in None frame")
            return []
            
        # Use the improved face analysis with multiple detection sizes
        faces = self.face_analyzer.get_with_multiple_sizes(frame, max_faces)
        
        # Filter out faces that are too small
        if min_face_size > 0:
            filtered_faces = []
            for face in faces:
                bbox = face.bbox.astype(int)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                if width >= min_face_size and height >= min_face_size:
                    filtered_faces.append(face)
                else:
                    self.logger.debug(f"Filtered out face that's too small: {width}x{height} pixels")
            
            faces = filtered_faces
        
        self.logger.info(f"Detected {len(faces)} faces in frame")
        return faces
    
    def find_person_by_face(self, face_image: np.ndarray, known_faces: Dict, threshold: float = None) -> Tuple[str, float]:
        """
        Find a person by their face.
        
        Args:
            face_image: Face image
            known_faces: Dictionary of known faces
            threshold: Optional threshold (overrides default)
            
        Returns:
            Tuple[str, float]: Person ID and similarity score
        """
        if threshold is None:
            threshold = self.recognition_threshold
            
        faces = self.detect_faces(face_image, max_faces=1)
        
        if not faces:
            self.logger.warning("No faces detected in the image")
            return None, 0.0
            
        face = faces[0]
        face_embedding = face.embedding
        
        person_id, similarity = self.recognize_face(known_faces, face_embedding)
        
        if person_id and similarity > threshold:
            return person_id, similarity
        else:
            return None, similarity
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate the similarity between two face embeddings.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
            
        try:
            # Normalize the embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            embedding1 = embedding1 / norm1
            embedding2 = embedding2 / norm2
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2)
            
            # Convert to a 0-1 range
            similarity = max(0, min(1, (similarity + 1) / 2))
            
            return similarity
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0 