"""
Core module for facial recognition.
Contains the main FacialRecognitionModule class that integrates all components.
"""

import os
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any

from modules.facial_recognition.improved_analysis import ImprovedFaceAnalysis
from modules.facial_recognition.persistence import FacialRecognitionPersistence
from modules.facial_recognition.recognition import FaceRecognition
from modules.facial_recognition.integration import FaceVideoIntegration

class FacialRecognitionModule:
    """
    Main class for facial recognition that integrates all components.
    """
    
    def __init__(self, 
                recognition_threshold: float = 0.5,
                face_db_path: str = "data/face_db",
                model_name: str = "buffalo_l",
                detection_size: Tuple[int, int] = (640, 640),
                max_faces: int = 5):
        """
        Initialize facial recognition module.
        
        Args:
            recognition_threshold: Threshold for face recognition (0.0-1.0)
            face_db_path: Path to face database
            model_name: Model name for face analysis
            detection_size: Size for face detection
            max_faces: Maximum number of faces to detect
        """
        self.logger = logging.getLogger("facial_recognition")
        self.logger.info(f"Initializing FacialRecognitionModule with threshold {recognition_threshold}")
        
        # Initialize face analyzer
        try:
            self.face_analyzer = ImprovedFaceAnalysis(
                name=model_name,
                root="~/.insightface",
                providers=['CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=detection_size)
            self.logger.info(f"Initialized face analyzer with model {model_name}")
        except Exception as e:
            self.logger.error(f"Error initializing face analyzer: {e}")
            raise
        
        # Initialize components
        self.persistence = FacialRecognitionPersistence(face_db_path)
        self.recognition = FaceRecognition(
            face_analyzer=self.face_analyzer,
            recognition_threshold=recognition_threshold
        )
        self.integration = FaceVideoIntegration(
            face_recognizer=self.recognition,
            persistence_manager=self.persistence
        )
        
        # Set parameters
        self.max_faces = max_faces
        self.recognition_threshold = recognition_threshold
        
        # Load face database
        self.load_data()
        
    def load_data(self) -> None:
        """
        Load all necessary data from disk.
        """
        try:
            # Load known faces
            self.logger.info("Loading face database...")
            self.persistence.load_face_db()
            
            # Load identity mappings
            self.logger.info("Loading identity mappings...")
            self.persistence.load_identity_mappings()
            
            # Load persistent identities
            self.logger.info("Loading persistent identities...")
            self.persistence.load_persistent_identities()
            
            # Load face galleries
            self.logger.info("Loading face galleries...")
            self.persistence.load_face_galleries()
            
            self.logger.info(f"Loaded {len(self.persistence.known_faces)} known faces")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
            
    def save_data(self) -> None:
        """
        Save all data to disk.
        """
        try:
            # Save known faces
            self.logger.info("Saving face database...")
            self.persistence.save_face_db()
            
            # Save identity mappings
            self.logger.info("Saving identity mappings...")
            self.persistence.save_identity_mappings()
            
            # Save persistent identities
            self.logger.info("Saving persistent identities...")
            self.persistence.save_persistent_identities()
            
            self.logger.info("All data saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            raise
            
    def detect_faces(self, image: np.ndarray) -> List:
        """
        Detect faces in an image.
        
        Args:
            image: Input image
            
        Returns:
            List: Detected faces
        """
        return self.recognition.detect_faces(image)
        
    def recognize_face(self, face_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize a face from its embedding.
        
        Args:
            face_embedding: Face embedding vector
            
        Returns:
            Tuple[str, float]: Person ID and similarity score
        """
        return self.recognition.recognize_face(
            self.persistence.known_faces,
            face_embedding
        )
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a video frame for face detection and recognition.
        
        Args:
            frame: Video frame
            
        Returns:
            Dict: Processing results
        """
        return self.integration.process_video_frame(frame)
        
    def add_face(self, 
                person_id: str, 
                face_embedding: np.ndarray, 
                face_image: Optional[np.ndarray] = None) -> bool:
        """
        Add a face to the database.
        
        Args:
            person_id: Person identifier
            face_embedding: Face embedding vector
            face_image: Optional face image
            
        Returns:
            bool: Success flag
        """
        try:
            # Check if face already exists
            if person_id in self.persistence.known_faces:
                # Append to existing embeddings
                self.persistence.known_faces[person_id].append(face_embedding)
                self.logger.info(f"Added new embedding for {person_id}")
            else:
                # Create new entry
                self.persistence.known_faces[person_id] = [face_embedding]
                self.logger.info(f"Added new person {person_id} to face database")
                
            # Save face image if provided
            if face_image is not None:
                self.persistence.save_face_image(person_id, face_image)
                
            # Save face database
            self.persistence.save_face_db()
            
            return True
        except Exception as e:
            self.logger.error(f"Error adding face: {e}")
            return False
            
    def remove_face(self, person_id: str) -> bool:
        """
        Remove a person from the face database.
        
        Args:
            person_id: Person identifier
            
        Returns:
            bool: Success flag
        """
        try:
            if person_id in self.persistence.known_faces:
                del self.persistence.known_faces[person_id]
                self.logger.info(f"Removed person {person_id} from face database")
                
                # Save face database
                self.persistence.save_face_db()
                
                return True
            else:
                self.logger.warning(f"Person {person_id} not found in face database")
                return False
        except Exception as e:
            self.logger.error(f"Error removing face: {e}")
            return False
            
    def set_recognition_threshold(self, threshold: float) -> None:
        """
        Set the recognition threshold.
        
        Args:
            threshold: Recognition threshold (0.0-1.0)
        """
        self.recognition_threshold = max(0.0, min(1.0, threshold))
        self.recognition.recognition_threshold = self.recognition_threshold
        self.logger.info(f"Set recognition threshold to {self.recognition_threshold}")
        
    def process_conversation_with_video(self, 
                                        video_file: str, 
                                        diarized_segments: List[Dict],
                                        output_dir: str,
                                        knowledge_base: Dict = None) -> List[Dict]:
        """
        Process a conversation with video, associating speakers with faces.
        
        Args:
            video_file: Path to video file
            diarized_segments: List of diarized speech segments
            output_dir: Directory for output files
            knowledge_base: Optional knowledge base
            
        Returns:
            List[Dict]: Enhanced segments with face information
        """
        return self.integration.process_conversation_with_video(
            video_file,
            diarized_segments,
            output_dir,
            knowledge_base
        )
        
    def save_debug_face_image(self, frame: np.ndarray, faces: list) -> str:
        """
        Save a debug image with face bounding boxes.
        
        Args:
            frame: Input frame
            faces: List of detected faces
            
        Returns:
            str: Path to saved image
        """
        return self.integration.save_debug_face_image(frame, faces)
        
    def associate_speaker_with_face(self, speaker_id: str, person_id: str) -> None:
        """
        Associate a speaker ID with a face/person.
        
        Args:
            speaker_id: Speaker ID from diarization
            person_id: Person identifier
        """
        self.integration.associate_speaker_with_face(speaker_id, person_id)
        
    def get_person_from_speaker(self, speaker_id: str) -> str:
        """
        Get person associated with a speaker.
        
        Args:
            speaker_id: Speaker ID from diarization
            
        Returns:
            str: Person identifier
        """
        return self.integration.get_person_from_speaker(speaker_id)
        
    def clear_current_user(self) -> None:
        """
        Clear the current user tracking.
        """
        self.recognition.update_current_face(None, None)
        self.persistence.clear_current_user_id()
        
    def release(self) -> None:
        """
        Release resources and save data.
        """
        self.save_data()
        self.logger.info("FacialRecognitionModule released") 