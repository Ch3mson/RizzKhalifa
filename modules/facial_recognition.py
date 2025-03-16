#!/usr/bin/env python3

import os
import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Set
import pickle
import logging
from pathlib import Path
import json

from insightface.app import FaceAnalysis

# Local imports
from modules.config import SAMPLE_RATE

# Global variable for face recheck interval (in seconds)
FACE_RECHECK_INTERVAL = 300  # Default: check every 5 minutes

class ImprovedFaceAnalysis(FaceAnalysis):
    """
    Extension of InsightFace's FaceAnalysis that offers improved face detection capabilities.
    This class provides better handling of detection sizes and implements adaptive detection.
    """
    
    def get_with_multiple_sizes(self, img, max_num=0, sizes=None):
        """
        Attempts to detect faces using multiple detection sizes.
        
        Args:
            img: Input image
            max_num: Maximum number of faces to detect (0 for all)
            sizes: List of detection sizes to try [(width, height)]
            
        Returns:
            List of detected faces
        """
        if sizes is None:
            sizes = [(640, 640), (320, 320), (480, 480), (720, 720), (960, 960)]
        
        print(f"Trying to detect faces with multiple detection sizes: {sizes}")
        faces = None
        
        for det_size in sizes:
            try:
                # Set detection size for this attempt
                if hasattr(self.det_model, "input_size"):
                    self.det_model.input_size = det_size
                
                # Try detection with current size
                faces = self.get(img, max_num)
                if faces and len(faces) > 0:
                    print(f"Successfully detected {len(faces)} faces with detection size {det_size}")
                    return faces
            except Exception as e:
                print(f"Error with detection size {det_size}: {e}")
                continue
        
        # If no faces found with any size, return empty list
        if not faces or len(faces) == 0:
            print("No faces detected with any detection size")
            return []
        
        return faces

class FacialRecognitionModule:
    """
    Module for facial recognition that works alongside speaker diarization.
    
    This module uses InsightFace to recognize faces in video frames and 
    associates them with speakers identified by the diarization system.
    """
    
    def __init__(self, 
                recognition_threshold: float = 0.5, 
                face_db_path: str = None,
                model_name: str = 'buffalo_l'):
        """
        Initialize the facial recognition module.
        
        Args:
            recognition_threshold: Threshold for face recognition (0.0-1.0)
            face_db_path: Path to save face database (now only used for speaker mapping)
            model_name: InsightFace model name
        """
        # Set up logging
        self.logger = logging.getLogger("facial_recognition")
        
        # Set recognition threshold
        self.recognition_threshold = recognition_threshold
        print(f"Using face recognition threshold: {self.recognition_threshold}")
        
        # Initialize InsightFace
        try:
            print(f"Initializing InsightFace with model: {model_name}")
            # Use recommended providers with fallback to CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            # Use our improved face analysis class
            self.app = ImprovedFaceAnalysis(name=model_name, providers=providers)
            # Don't prepare yet - we'll prepare with different detection sizes as needed
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.logger.info(f"Initialized InsightFace with model {model_name}")
            print(f"Successfully initialized InsightFace model: {model_name}")
        except Exception as e:
            self.logger.error(f"Error initializing InsightFace: {e}")
            print(f"Error initializing InsightFace: {e}")
            raise
        
        # Get project directory
        project_dir = os.getcwd()
        
        if not face_db_path:
            # Create a separate system directory for face database
            system_dir = os.path.join(project_dir, "conversations", "system_data")
            os.makedirs(system_dir, exist_ok=True)
            face_db_path = os.path.join(system_dir, "speaker_mapping.json")
        
        self.face_db_path = face_db_path
        self.known_faces = {}  # name -> embedding
        self.speaker_face_mapping = {}  # speaker_id -> person_name
        self.face_count = 0  # Counter for auto-naming unknown faces
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.face_db_path), exist_ok=True)
        
        # Track the current face for this conversation
        self.current_face_name = None
        self.current_face_embedding = None
        self.last_face_check_time = 0
        
        # Load existing face database by scanning directories
        self._load_face_db()
        
        # Load face galleries from image files
        self._load_face_galleries()
        
        # Load persistent identities for cross-session recognition
        self._load_persistent_identities()
        
        # Initialize identity mappings
        self.identity_mappings = {}
        self._load_identity_mappings()
    
    def _load_face_db(self) -> None:
        """
        Load the face database by scanning conversation directories for face images.
        Also loads speaker mapping from JSON if available.
        """
        # First, load any speaker mapping data from JSON
        mapping_path = os.path.join(os.getcwd(), "conversations", "system_data", "speaker_mapping.json")
        if os.path.exists(mapping_path):
            try:
                with open(mapping_path, 'r') as f:
                    data = json.load(f)
                    self.speaker_face_mapping = data.get('mapping', {})
                    self.face_count = data.get('face_count', 0)
                self.logger.info(f"Loaded speaker mapping with {len(self.speaker_face_mapping)} entries")
            except Exception as e:
                self.logger.error(f"Error loading speaker mapping: {e}")
                self.speaker_face_mapping = {}
        
        # Now scan directories for face images
        try:
            base_dir = os.path.join(os.getcwd(), "conversations")
            if not os.path.exists(base_dir):
                self.logger.info("No conversations directory found. Creating new database.")
                os.makedirs(base_dir, exist_ok=True)
                return
            
            # Look for person directories with face.jpg files
            person_count = 0
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                face_path = os.path.join(item_path, "face.jpg")
                
                # If this is a directory with a face.jpg file
                if os.path.isdir(item_path) and os.path.exists(face_path):
                    try:
                        # Load the face image
                        face_image = cv2.imread(face_path)
                        if face_image is None:
                            self.logger.warning(f"Could not read face image for {item}")
                            continue
                        
                        # Detect and get face embedding
                        faces = self.app.get(face_image)
                        if not faces:
                            self.logger.warning(f"No face detected in saved image for {item}")
                            continue
                        
                        # Use the largest face if multiple are detected
                        if len(faces) > 1:
                            # Sort by face box area
                            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                        
                        # Store the face embedding
                        face_embedding = faces[0].normed_embedding
                        self.known_faces[item] = face_embedding
                        person_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error processing face image for {item}: {e}")
            
            self.logger.info(f"Loaded {person_count} faces from image files")
        except Exception as e:
            self.logger.error(f"Error scanning for face images: {e}")
            self.known_faces = {}
    
    def _save_face_db(self) -> None:
        """Save the face database metadata to disk using JSON."""
        try:
            # Create system directory if it doesn't exist
            system_dir = os.path.join(os.getcwd(), "conversations", "system_data")
            os.makedirs(system_dir, exist_ok=True)
            
            # Save only the mapping information, not the embeddings
            mapping_path = os.path.join(system_dir, "speaker_mapping.json")
            with open(mapping_path, 'w') as f:
                json.dump({
                    'mapping': self.speaker_face_mapping,
                    'face_count': self.face_count
                }, f, indent=2)
            
            self.logger.info(f"Saved speaker mapping with {len(self.speaker_face_mapping)} entries")
        except Exception as e:
            self.logger.error(f"Error saving face database metadata: {e}")
            
    def set_recheck_interval(self, seconds: int) -> None:
        """
        Set the interval for rechecking if it's the same face.
        
        Args:
            seconds: Interval in seconds between face checks
        """
        global FACE_RECHECK_INTERVAL
        FACE_RECHECK_INTERVAL = seconds
        self.logger.info(f"Face recheck interval set to {FACE_RECHECK_INTERVAL} seconds")
        
    def get_recheck_interval(self) -> int:
        """
        Get the interval for rechecking faces in seconds.
        
        Returns:
            int: Interval in seconds
        """
        return FACE_RECHECK_INTERVAL
    
    def add_face(self, name: str, face_image: np.ndarray) -> bool:
        """
        Add a new face to the database with a name.
        
        Args:
            name: Name of the person
            face_image: Image containing a face
            
        Returns:
            bool: True if face was added successfully, False otherwise
        """
        try:
            # Detect and get face embedding
            faces = self.app.get(face_image)
            
            if not faces:
                self.logger.warning(f"No face detected in the provided image for {name}")
                return False
            
            # Use the largest face if multiple are detected
            if len(faces) > 1:
                self.logger.info(f"Multiple faces detected, using the largest one for {name}")
                # Sort by face box area
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            
            # Store the face embedding
            face_embedding = faces[0].normed_embedding
            self.known_faces[name] = face_embedding
            
            # Save the updated database
            self._save_face_db()
            self.logger.info(f"Added face for {name} to database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding face: {e}")
            return False
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Recognize a face in an image.
        
        Args:
            face_image: Image containing a face
            
        Returns:
            Tuple[str, float]: (person_name, confidence) or ("unknown", 0.0) if no match
        """
        try:
            # Detect and get face embedding
            faces = self.app.get(face_image)
            
            if not faces:
                return "unknown", 0.0
            
            # Use the largest face if multiple are detected
            if len(faces) > 1:
                # Sort by face box area
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            
            face_embedding = faces[0].normed_embedding
            
            # Find the closest match
            best_match = "unknown"
            best_similarity = 0.0
            
            for name, known_embedding in self.known_faces.items():
                # Calculate cosine similarity
                similarity = np.dot(face_embedding, known_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
            
            # Check if similarity exceeds threshold
            if best_similarity >= self.recognition_threshold:
                return best_match, float(best_similarity)
            else:
                return "unknown", float(best_similarity)
                
        except Exception as e:
            self.logger.error(f"Error recognizing face: {e}")
            return "unknown", 0.0
    
    def process_video_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a video frame to detect and recognize faces.
        
        Args:
            frame: Video frame as numpy array
            
        Returns:
            Dict: Results with face detection and recognition information
        """
        try:
            # Detect faces in the frame
            faces = self.app.get(frame)
            
            results = []
            
            if not faces:
                self.logger.debug("No faces detected in frame")
                return results
            
            # Process each detected face
            for i, face in enumerate(faces):
                try:
                    # Get face embedding for recognition
                    face_embedding = face.embedding
                    bbox = face.bbox
                    
                    # Extract face image for saving
                    x1, y1, x2, y2 = [int(b) for b in bbox]
                    # Add some margin
                    margin = 20
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame.shape[1], x2 + margin)
                    y2 = min(frame.shape[0], y2 + margin)
                    face_image = frame[y1:y2, x1:x2]
                    
                    # Check if this is a known face
                    name = None
                    confidence = 0.0
                    
                    if self.known_faces:
                        # First try to get a persistent identity
                        name = self.get_persistent_identity(face_embedding)
                        
                        # Check if this name maps to a canonical identity
                        if hasattr(self, 'identity_mappings') and name in self.identity_mappings:
                            canonical_name = self.identity_mappings[name]
                            self.logger.info(f"Mapped {name} to canonical identity: {canonical_name}")
                            name = canonical_name
                        
                        if name != "unknown":
                            confidence = 1.0  # We're confident in our persistent identity system
                            self.logger.info(f"Recognized persistent identity: {name}")
                            
                            # Update the embedding for this person
                            self.known_faces[name] = face_embedding
                            
                            # Also add to the person's face gallery
                            self.add_to_face_gallery(name, face_embedding)
                            
                            # Save the updated database
                            self._save_face_db()
                            
                            # Update the current face and explicitly save the face image
                            self.update_current_face(name, face_embedding, face_image)
                    
                    # If no match found, create a new person
                    if not name:
                        # Try to find an existing folder for this face
                        existing_person = self.find_existing_person_folder(face_image)
                        if existing_person:
                            # Use the existing person folder
                            person_id = existing_person
                            self.logger.info(f"Using existing person folder: {person_id}")
                        else:
                            # Create a new timestamp-based ID
                            current_time = int(time.time())
                            person_id = f"Person_{current_time}"
                            self.logger.info(f"Created new person folder: {person_id}")
                        
                        # Create a consistent name with timestamp
                        new_name = f"Person_{current_time}"
                        # Only increment if we're creating different faces
                        if len(self.known_faces) > 0:
                            self.face_count += 1
                            new_name = f"Person_{current_time}_{self.face_count}"
                            
                        self.known_faces[new_name] = face_embedding
                        self._save_face_db()
                        name, confidence = new_name, 1.0  # New face is 100% match to itself
                        self.logger.info(f"Added new face automatically: {new_name}")
                        
                        # Save the face image for the new person - explicitly call with face image
                        self.update_current_face(new_name, face_embedding, face_image)
                    
                    # Add result
                    results.append({
                        "id": i,
                        "bbox": bbox.tolist(),
                        "name": name,
                        "confidence": confidence,
                        "embedding": face_embedding
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error processing face {i}: {e}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing video frame: {e}")
        
        return []
    
    def associate_speaker_with_face(self, speaker_id: str, person_name: str) -> None:
        """
        Associate a speaker ID from diarization with a person name from facial recognition.
        
        Args:
            speaker_id: Speaker ID from diarization (e.g., "SPEAKER_1")
            person_name: Person name from facial recognition
        """
        self.speaker_face_mapping[speaker_id] = person_name
        self._save_face_db()
        self.logger.info(f"Associated speaker {speaker_id} with person {person_name}")
    
    def get_person_from_speaker(self, speaker_id: str) -> str:
        """
        Get person name associated with a speaker ID.
        
        Args:
            speaker_id: Speaker ID from diarization
            
        Returns:
            str: Person name or speaker_id if no association exists
        """
        return self.speaker_face_mapping.get(speaker_id, speaker_id)
    
    def process_conversation_with_video(self, 
                                      video_file: str, 
                                      diarized_segments: List[Dict],
                                      output_dir: str,
                                      knowledge_base: Dict = None,
                                      workflow_state: Dict = None,
                                      use_max_resolution: bool = True,
                                      camera_resolution: Tuple[int, int] = (1920, 1080)) -> List[Dict]:
        """
        Process a conversation with video, associating speakers with faces.
        Simplified to capture just one face per conversation and wait until one is found.
        
        Args:
            video_file: Path to video file
            diarized_segments: List of diarized speech segments with speaker labels
            output_dir: Directory to save conversation data
            knowledge_base: Optional knowledge base to save with the conversation
            workflow_state: Optional full workflow state for additional context
            use_max_resolution: Whether to attempt to use maximum camera resolution
            camera_resolution: Resolution to set for the camera (width, height) if not using max
            
        Returns:
            List[Dict]: Updated diarized segments with person names
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                self.logger.error(f"Could not open video file: {video_file}")
                return diarized_segments
            
            # Set camera resolution if this is a camera device
            if str(video_file).isdigit() or video_file in [0, 1]:
                if use_max_resolution:
                    # Try common high resolutions in descending order
                    # The camera will use the highest supported resolution
                    resolutions_to_try = [
                        (4096, 2160),  # 4K UHD
                        (3840, 2160),  # 4K UHD (16:9)
                        (2560, 1440),  # QHD
                        (1920, 1080),  # Full HD
                        (1280, 720),   # HD
                    ]
                    
                    # Store original properties to restore if needed
                    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # Try each resolution
                    best_width, best_height = original_width, original_height
                    
                    for width, height in resolutions_to_try:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        
                        # Check what resolution was actually set
                        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        # If we got a higher resolution than before, keep it
                        if actual_width * actual_height > best_width * best_height:
                            best_width, best_height = actual_width, actual_height
                    
                    # Set the best resolution we found
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_height)
                    self.logger.info(f"Set camera to best available resolution: {best_width}x{best_height}")
                else:
                    # Use the specified resolution
                    width, height = camera_resolution
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    self.logger.info(f"Set camera resolution to {width}x{height}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.logger.info(f"Video properties: {actual_width}x{actual_height} at {fps} FPS, {total_frames} frames")
            
            # Initialize variables for face detection
            face_detected = False
            detected_person = None
            max_attempts = 10  # Maximum number of attempts to find a face
            attempts = 0
            
            # Keep trying until we find a face or reach max attempts
            face_detections = []  # Track multiple detections for consistency
            required_detections = 3  # Require multiple consistent detections
            while len(face_detections) < required_detections and attempts < max_attempts:
                attempts += 1
                
                # Get a frame from the video (try different positions)
                if attempts == 1:
                    # First try the middle
                    frame_pos = total_frames // 2
                elif attempts == 2:
                    # Then try 1/3 of the way through
                    frame_pos = total_frames // 3
                elif attempts == 3:
                    # Then try 2/3 of the way through
                    frame_pos = (total_frames * 2) // 3
                else:
                    # Then try random positions
                    frame_pos = int(np.random.uniform(0, total_frames))
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if not ret:
                    self.logger.warning(f"Failed to read frame (attempt {attempts}/{max_attempts})")
                    continue
                
                # Process the frame to detect faces
                face_results = self.process_video_frame(frame)
                
                if face_results:
                    # We found at least one face
                    face_info = face_results[0]
                    detected_person = face_info["name"]
                    face_detections.append((detected_person, face_info["embedding"], frame))
                    self.logger.info(f"Detected person: {detected_person} (detection {len(face_detections)}/{required_detections})")
            
            # After collecting detections, check for consistency
            if face_detections:
                # Check if we have enough consistent detections
                if len(face_detections) >= required_detections * 0.7:  # At least 70% of required
                    # Count occurrences of each person
                    person_counts = {}
                    for person, _, _ in face_detections:
                        person_counts[person] = person_counts.get(person, 0) + 1
                    
                    # Find the most common person
                    most_common_person = max(person_counts.items(), key=lambda x: x[1])[0]
                    
                    # Use the most common person
                    face_detected = True
                    detected_person = self.standardize_person_name(most_common_person, output_dir)
                    self.logger.info(f"Consistently detected person: {detected_person}")
                    
                    # Find a good frame for this person
                    best_frame = None
                    best_embedding = None
                    for person, embedding, frame in face_detections:
                        if person == detected_person:
                            best_frame = frame
                            best_embedding = embedding
                            break
            
            # If no face was detected after multiple attempts
            if not face_detected:
                self.logger.warning("No face detected in video after multiple attempts")
                
                # Create a timestamp-based unknown person ID
                current_time = int(time.time())
                unknown_id = f"Person_{current_time}"
                
                # Create an unknown person entry
                for segment in diarized_segments:
                    segment["person"] = unknown_id
                
                # Prepare the conversation data
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                conversation_data = {
                    "timestamp": timestamp,
                    "segments": diarized_segments
                }
                
                # Add the knowledge base if provided
                if knowledge_base:
                    conversation_data["knowledge_base"] = knowledge_base
                
                # Add the entire workflow state if provided
                if workflow_state:
                    conversation_data["workflow_state"] = workflow_state
                
                # Save the conversation data
                self.save_conversation_for_person(
                    person_name=unknown_id,
                    conversation_data=conversation_data,
                    base_dir=output_dir
                )
            
            # Release video capture
            cap.release()
            return diarized_segments
            
        except Exception as e:
            self.logger.error(f"Error processing conversation with video: {e}")
            import traceback
            traceback.print_exc()
            return diarized_segments
    
    def save_conversation_for_person(self, 
                                    person_name: str, 
                                    conversation_data: Dict,
                                    base_dir: str = None) -> bool:
        """
        This method has been deactivated since we no longer save to Person directories.
        
        Args:
            person_name: Name of the person
            conversation_data: Conversation data to save (including knowledge base)
            base_dir: Base directory to save conversations
            
        Returns:
            bool: Always returns True (but doesn't actually save anything)
        """
        self.logger.info("save_conversation_for_person is deactivated - no longer saving to Person directories")
        return True
    
    def capture_from_webcam(self, name: str, duration: int = 5) -> bool:
        """
        Capture a person's face from webcam and add to database.
        
        Args:
            name: Name of the person
            duration: Duration in seconds to capture face
            
        Returns:
            bool: True if face was captured and added successfully
        """
        try:
            # Open webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.logger.error("Could not open webcam")
                return False
            
            self.logger.info(f"Capturing face for {name}. Please look at the camera...")
            
            # Wait for the camera to initialize
            time.sleep(1)
            
            # Capture frames for the specified duration
            start_time = time.time()
            frames = []
            
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Add frame to list
                frames.append(frame)
                
                # Display frame
                cv2.imshow("Capturing face...", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Release webcam and destroy window
            cap.release()
            cv2.destroyAllWindows()
            
            if not frames:
                self.logger.error("No frames captured")
                return False
            
            # Process frames to find the best face
            best_face_frame = None
            best_face_size = 0
            
            for frame in frames:
                faces = self.app.get(frame)
                if faces:
                    # Find the largest face
                    for face in faces:
                        bbox = face.bbox
                        face_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if face_size > best_face_size:
                            best_face_size = face_size
                            best_face_frame = frame
            
            if best_face_frame is None:
                self.logger.error("No face detected during capture")
                return False
            
            # Add the best face to the database
            return self.add_face(name, best_face_frame)
            
        except Exception as e:
            self.logger.error(f"Error capturing face from webcam: {e}")
            return False
    
    def should_recheck_face(self) -> bool:
        """
        Determine if it's time to recheck the face based on the interval.
        
        Returns:
            bool: True if it's time to recheck, False otherwise
        """
        current_time = time.time()
        if current_time - self.last_face_check_time >= FACE_RECHECK_INTERVAL:
            return True
        return False

    def update_current_face(self, face_name: str, face_embedding: np.ndarray, face_image: np.ndarray = None) -> None:
        """
        Update the current face for this conversation session.
        
        Args:
            face_name: Name of the person
            face_embedding: Face embedding vector
            face_image: Optional face image to save
        """
        # Ensure we're using timestamp-based names
        if not face_name.startswith("Person_") and not face_name.startswith("Unknown_"):
            if not hasattr(self, '_session_timestamp'):
                self._session_timestamp = int(time.time())
            face_name = f"Person_{self._session_timestamp}"
        
        self.current_face_name = face_name
        self.current_face_embedding = face_embedding
        self.last_face_check_time = time.time()
        
        # Also make sure it's saved in the known faces
        if face_name not in self.known_faces:
            self.known_faces[face_name] = face_embedding
            self.logger.info(f"Added new face '{face_name}' to known faces database")
        else:
            # Update the embedding to improve future recognition
            # Use a weighted average to gradually adapt to changes in appearance
            existing_embedding = self.known_faces[face_name]
            updated_embedding = 0.7 * existing_embedding + 0.3 * face_embedding
            # Normalize the embedding
            updated_embedding = updated_embedding / np.linalg.norm(updated_embedding)
            self.known_faces[face_name] = updated_embedding
            self.logger.debug(f"Updated face embedding for '{face_name}'")
        
        # Save face image if provided - this is now our primary storage method
        if face_image is not None:
            self._save_face_image(face_name, face_image)
        
        # Save the speaker mapping
        self._save_face_db()

    def _save_face_image(self, person_name: str, face_image: np.ndarray) -> None:
        """
        Save a face image for a person. This is now the primary method for storing face data.
        
        Args:
            person_name: Name of the person
            face_image: Face image to save
        """
        try:
            # Create person directory if it doesn't exist
            base_dir = os.path.join(os.getcwd(), "conversations")
            person_dir = os.path.join(base_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            # Save face image
            face_path = os.path.join(person_dir, "face.jpg")
            cv2.imwrite(face_path, face_image)
            
            # Also save a timestamped version for history
            timestamp = int(time.time())
            history_face_path = os.path.join(person_dir, f"face_{timestamp}.jpg")
            cv2.imwrite(history_face_path, face_image)
            
            self.logger.info(f"Saved face image for {person_name}")
        except Exception as e:
            self.logger.error(f"Error saving face image: {e}")

    def is_same_face(self, face_embedding: np.ndarray, threshold: float = 0.4) -> bool:
        """
        Check if a face embedding matches the current face.
        Lower threshold = stricter matching (fewer false positives)
        
        Args:
            face_embedding: Face embedding to compare
            threshold: Similarity threshold (lower = stricter)
            
        Returns:
            bool: True if it's the same face, False otherwise
        """
        if not hasattr(self, 'current_face_embedding') or self.current_face_embedding is None:
            return False
        
        # Calculate cosine similarity
        similarity = np.dot(self.current_face_embedding, face_embedding) / (
            np.linalg.norm(self.current_face_embedding) * np.linalg.norm(face_embedding))
        
        self.logger.info(f"Face similarity score: {similarity:.4f} (threshold: {threshold})")
        return similarity > threshold

    def migrate_existing_data(self, base_dir: str = None) -> bool:
        """
        Migrate existing conversation data to the new person### format.
        
        Args:
            base_dir: Base directory containing conversations
            
        Returns:
            bool: True if migration was successful
        """
        try:
            # Set up directory in the current project folder
            base_dir = base_dir or os.path.join(os.getcwd(), "conversations")
            if not os.path.exists(base_dir):
                self.logger.warning(f"Conversations directory {base_dir} does not exist. Nothing to migrate.")
                return False
            
            # Get all directories that need migration (Person_* or Unknown_*)
            dirs_to_migrate = []
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path) and (item.startswith("Person_") or item.startswith("Unknown_")):
                    dirs_to_migrate.append(item)
            
            if not dirs_to_migrate:
                self.logger.info("No directories need migration.")
                return True
            
            self.logger.info(f"Found {len(dirs_to_migrate)} directories to migrate")
            
            # Create mapping for old to new directory names
            migration_map = {}
            next_person_number = 1
            
            # First pass: create the mapping
            for old_dir in dirs_to_migrate:
                new_dir = f"person{next_person_number:03d}"
                migration_map[old_dir] = new_dir
                next_person_number += 1
            
            # Second pass: perform the migration
            for old_dir, new_dir in migration_map.items():
                old_path = os.path.join(base_dir, old_dir)
                new_path = os.path.join(base_dir, new_dir)
                
                # Create the new directory
                os.makedirs(new_path, exist_ok=True)
                
                # Copy all files
                for item in os.listdir(old_path):
                    old_item_path = os.path.join(old_path, item)
                    new_item_path = os.path.join(new_path, item)
                    
                    # Special handling for conversation files
                    if item.startswith("conversation_") and item.endswith(".txt"):
                        # Copy the file
                        with open(old_item_path, 'r') as src, open(new_item_path, 'w') as dst:
                            content = src.read()
                            # Replace old person name with new one in content
                            content = content.replace(old_dir.upper(), new_dir.upper())
                            dst.write(content)
                            
                        # Also append to conversation_history.txt
                        history_path = os.path.join(new_path, "conversation_history.txt")
                        history_exists = os.path.exists(history_path)
                        
                        with open(history_path, 'a' if history_exists else 'w') as f:
                            if not history_exists:
                                f.write(f"CONVERSATION HISTORY FOR {new_dir.upper()}\n")
                                f.write("="*80 + "\n\n")
                            
                            with open(old_item_path, 'r') as src:
                                # Skip the header lines
                                lines = src.readlines()
                                if len(lines) > 3:  # Skip title and separator
                                    f.write(f"\n--- MIGRATED FROM {old_dir} ---\n\n")
                                    f.writelines(lines[3:])  # Write content after header
                    else:
                        # For other files, just copy them
                        if os.path.isfile(old_item_path):
                            with open(old_item_path, 'rb') as src, open(new_item_path, 'wb') as dst:
                                dst.write(src.read())
                
                # After successful migration, you can optionally remove the old directory
                # import shutil
                # shutil.rmtree(old_path)
                
                self.logger.info(f"Migrated {old_dir} to {new_dir}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error migrating data: {e}")
            import traceback
            traceback.print_exc()
            return False 

    def standardize_person_name(self, person_name: str, base_dir: str = None) -> str:
        """
        Keep timestamp-based names as they are.
        
        Args:
            person_name: Original person name (e.g., "Person_1742006765")
            base_dir: Base directory for conversations
            
        Returns:
            str: Original person name
        """
        # Always return the original timestamp-based name
        if person_name.startswith("Person_") or person_name.startswith("Unknown_"):
            return person_name
        
        # If it's already in person### format, convert it to a timestamp format
        if person_name.startswith("person") and person_name[6:].isdigit():
            # Create a timestamp-based name
            if not hasattr(self, '_session_timestamp'):
                self._session_timestamp = int(time.time())
            
            new_name = f"Person_{self._session_timestamp}"
            self.logger.info(f"Converting {person_name} to timestamp format: {new_name}")
            
            # Update the known_faces mapping
            if person_name in self.known_faces:
                self.known_faces[new_name] = self.known_faces[person_name]
                del self.known_faces[person_name]
                self._save_face_db()
            
            return new_name
        
        # For any other format, create a new timestamp-based name
        if not hasattr(self, '_session_timestamp'):
            self._session_timestamp = int(time.time())
        
        new_name = f"Person_{self._session_timestamp}"
        self.logger.info(f"Created timestamp-based name: {new_name} for {person_name}")
        
        return new_name 

    def add_to_face_gallery(self, person_name: str, face_embedding: np.ndarray) -> None:
        """
        Add a face embedding to a person's gallery for better recognition.
        
        Args:
            person_name: Name of the person
            face_embedding: Face embedding to add to gallery
        """
        # Initialize gallery if it doesn't exist
        if not hasattr(self, 'face_galleries'):
            self.face_galleries = {}
        
        # Create gallery for this person if it doesn't exist
        if person_name not in self.face_galleries:
            self.face_galleries[person_name] = []
        
        # Add embedding to gallery (limit to 5 most recent)
        self.face_galleries[person_name].append(face_embedding)
        if len(self.face_galleries[person_name]) > 5:
            self.face_galleries[person_name].pop(0)  # Remove oldest
        
        # Save the updated galleries
        self._save_face_galleries()
        
    def _save_face_galleries(self) -> None:
        """
        Save face galleries as multiple JPG images instead of pickle files.
        Each person's gallery is a collection of timestamped face images.
        """
        # This is now handled by saving multiple timestamped face images
        # in the _save_face_image method
        pass

    def _load_face_galleries(self) -> None:
        """
        Load face galleries by scanning for multiple face images per person.
        """
        self.face_galleries = {}
        
        try:
            base_dir = os.path.join(os.getcwd(), "conversations")
            if not os.path.exists(base_dir):
                return
            
            # Look for person directories with multiple face_*.jpg files
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                
                # Skip if not a directory
                if not os.path.isdir(item_path):
                    continue
                
                # Find all face_*.jpg files AND the main face.jpg
                face_images = []
                for file in os.listdir(item_path):
                    if (file.startswith("face_") and file.endswith(".jpg")) or file == "face.jpg":
                        face_path = os.path.join(item_path, file)
                        face_images.append(face_path)
                
                # Process all face images (not just 5)
                if face_images:
                    gallery = []
                    
                    for face_path in face_images:
                        try:
                            # Load the face image
                            face_image = cv2.imread(face_path)
                            if face_image is None:
                                continue
                            
                            # Detect and get face embedding
                            faces = self.app.get(face_image)
                            if not faces:
                                continue
                            
                            # Use the largest face if multiple are detected
                            if len(faces) > 1:
                                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                            
                            # Add embedding to gallery
                            gallery.append(faces[0].normed_embedding)
                            
                        except Exception as e:
                            self.logger.error(f"Error processing gallery image {face_path}: {e}")
                    
                    if gallery:
                        self.face_galleries[item] = gallery
                        self.logger.info(f"Loaded {len(gallery)} face images for {item}")
            
            self.logger.info(f"Loaded face galleries for {len(self.face_galleries)} people")
        except Exception as e:
            self.logger.error(f"Error loading face galleries: {e}")

    def recognize_face_with_gallery(self, face_embedding: np.ndarray) -> Tuple[str, float]:
        """
        Recognize a face using both single embedding and gallery matching.
        
        Args:
            face_embedding: Face embedding to recognize
            
        Returns:
            Tuple[str, float]: (person_name, confidence)
        """
        best_match = "unknown"
        best_similarity = 0.0
        
        # First try the standard single-embedding matching
        for name, known_embedding in self.known_faces.items():
            similarity = np.dot(face_embedding, known_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        # If we have galleries, also try gallery matching
        if hasattr(self, 'face_galleries') and self.face_galleries:
            for name, gallery in self.face_galleries.items():
                if not gallery:
                    continue
                
                # Calculate similarity with each face in the gallery
                gallery_similarities = [np.dot(face_embedding, gallery_face) for gallery_face in gallery]
                
                # Use the highest similarity from the gallery
                gallery_best = max(gallery_similarities)
                
                # If gallery match is better, use it
                if gallery_best > best_similarity:
                    best_similarity = gallery_best
                    best_match = name
        
        # Lower threshold for gallery matching (0.25 instead of 0.3)
        if best_similarity >= 0.25:
            return best_match, float(best_similarity)
        else:
            return "unknown", float(best_similarity)

    def get_persistent_identity(self, face_embedding: np.ndarray) -> str:
        """
        Get a persistent identity for a face embedding by comparing with all known faces.
        This helps maintain the same person ID across different sessions.
        
        Args:
            face_embedding: Face embedding to identify
            
        Returns:
            str: Persistent person ID or a new ID if no match found
        """
        best_match = None
        best_similarity = 0.0
        
        # First check against all known faces in the database
        for name, known_embedding in self.known_faces.items():
            similarity = np.dot(face_embedding, known_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        # Then check against all face galleries for better matching
        if hasattr(self, 'face_galleries') and self.face_galleries:
            for name, gallery in self.face_galleries.items():
                if not gallery:
                    continue
                
                # Calculate similarity with each face in the gallery
                gallery_similarities = [np.dot(face_embedding, gallery_face) for gallery_face in gallery]
                
                # Use the highest similarity from the gallery
                gallery_best = max(gallery_similarities)
                
                # If gallery match is better, use it
                if gallery_best > best_similarity:
                    best_similarity = gallery_best
                    best_match = name
        
        # Lower the threshold for persistent identity (from 0.2 to 0.15)
        # This makes it more likely to recognize the same person across sessions
        if best_similarity >= 0.15 and best_match:
            self.logger.info(f"Found persistent identity: {best_match} with similarity {best_similarity:.2f}")
            return best_match
        
        # If no match found, create a new persistent ID
        if not hasattr(self, '_session_timestamp'):
            self._session_timestamp = int(time.time())
        
        new_id = f"Person_{self._session_timestamp}"
        self.logger.info(f"Created new persistent identity: {new_id}")
        return new_id

    def _save_persistent_identities(self) -> None:
        """
        Persistent identities are now maintained through the face images.
        No need for separate storage.
        """
        # This functionality is now handled by the face images themselves
        pass

    def _load_persistent_identities(self) -> None:
        """
        Persistent identities are now loaded through scanning face images.
        This is handled by _load_face_db().
        """
        # This functionality is now handled by _load_face_db()
        pass

    def find_person_by_face(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Find a person by comparing a face image with all saved face images.
        
        Args:
            face_image: Image containing a face
            
        Returns:
            Tuple[str, float]: (person_name, confidence) or ("unknown", 0.0) if no match
        """
        try:
            # Detect and get face embedding
            faces = self.app.get(face_image)
            
            if not faces:
                return "unknown", 0.0
            
            # Use the largest face if multiple are detected
            if len(faces) > 1:
                # Sort by face box area
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            
            face_embedding = faces[0].normed_embedding
            
            # First try the standard single-embedding matching
            best_match = "unknown"
            best_similarity = 0.0
            
            for name, known_embedding in self.known_faces.items():
                similarity = np.dot(face_embedding, known_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
            
            # If we have galleries, also try gallery matching
            if hasattr(self, 'face_galleries') and self.face_galleries:
                for name, gallery in self.face_galleries.items():
                    if not gallery:
                        continue
                    
                    # Calculate similarity with each face in the gallery
                    gallery_similarities = [np.dot(face_embedding, gallery_face) for gallery_face in gallery]
                    
                    # Use the highest similarity from the gallery
                    gallery_best = max(gallery_similarities)
                    
                    # If gallery match is better, use it
                    if gallery_best > best_similarity:
                        best_similarity = gallery_best
                        best_match = name
            
            # Check if similarity exceeds threshold
            if best_similarity >= self.recognition_threshold:
                return best_match, float(best_similarity)
            else:
                return "unknown", float(best_similarity)
            
        except Exception as e:
            self.logger.error(f"Error finding person by face: {e}")
            return "unknown", 0.0

    def _save_identity_mappings(self) -> None:
        """
        Save identity mappings to connect different Person_timestamp IDs 
        that belong to the same person.
        """
        try:
            # Create system directory if it doesn't exist
            system_dir = os.path.join(os.getcwd(), "conversations", "system_data")
            os.makedirs(system_dir, exist_ok=True)
            
            # Create a mapping file if we have multiple IDs for the same person
            if hasattr(self, 'identity_mappings'):
                mapping_path = os.path.join(system_dir, "identity_mappings.json")
                with open(mapping_path, 'w') as f:
                    json.dump(self.identity_mappings, f, indent=2)
                self.logger.info(f"Saved identity mappings")
        except Exception as e:
            self.logger.error(f"Error saving identity mappings: {e}")

    def _load_identity_mappings(self) -> None:
        """
        Load identity mappings that connect different Person_timestamp IDs.
        """
        self.identity_mappings = {}
        
        try:
            mapping_path = os.path.join(os.getcwd(), "conversations", "system_data", "identity_mappings.json")
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    self.identity_mappings = json.load(f)
                self.logger.info(f"Loaded identity mappings")
        except Exception as e:
            self.logger.error(f"Error loading identity mappings: {e}")

    def merge_person_folders(self, source_id: str, target_id: str) -> bool:
        """
        This method has been deactivated since we no longer use Person directories.
        """
        self.logger.info("merge_person_folders is deactivated - no longer using Person directories")
        return True

    def find_existing_person_folder(self, face_image: np.ndarray) -> str:
        """
        This method has been deactivated since we no longer use Person directories.
        """
        self.logger.info("find_existing_person_folder is deactivated - no longer using Person directories")
        return None

    def _save_debug_face_image(self, frame: np.ndarray, faces: list, prefix: str = "debug") -> str:
        """
        Draws bounding boxes around detected faces and saves the image for debugging.
        
        Args:
            frame: The original frame
            faces: List of detected faces from InsightFace
            prefix: Prefix for the debug image filename
            
        Returns:
            str: Path to the saved debug image
        """
        try:
            # Create a copy of the frame to draw on
            debug_img = frame.copy()
            
            # Draw bounding boxes and confidence scores for each face
            for i, face in enumerate(faces):
                bbox = face.bbox.astype(int)
                # Draw bounding box (rectangle)
                cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Put confidence score text
                confidence = f"Score: {face.det_score:.4f}"
                cv2.putText(debug_img, confidence, (bbox[0], bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw landmark points if available
                if hasattr(face, 'kps') and face.kps is not None:
                    for kp in face.kps:
                        kx, ky = kp
                        cv2.circle(debug_img, (int(kx), int(ky)), 2, (255, 0, 0), 2)
            
            # Save the debug image
            debug_dir = os.path.join(os.getcwd(), "temp_files", "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            debug_path = os.path.join(debug_dir, f"{prefix}_faces_{int(time.time())}.jpg")
            cv2.imwrite(debug_path, debug_img)
            
            print(f"Saved debug face image with {len(faces)} detected faces to: {debug_path}")
            return debug_path
        except Exception as e:
            print(f"Error saving debug face image: {e}")
            return None
    
    def manage_face_recognition(self, frame: np.ndarray) -> Tuple[str, bool]:
        """
        Main entry point for the new face recognition flow:
        1. Takes a single frame containing a face
        2. Saves it to a temporary file
        3. Compares it with existing faces in conversations/faces/
        4. If there's a match, returns the identifier and deletes the temp file
        5. If no match or no existing faces, adds the face with a unique identifier
        
        Args:
            frame: The captured frame containing a face
            
        Returns:
            Tuple of (face_identifier, is_new_face)
        """
        # Create the faces directory if it doesn't exist
        faces_dir = os.path.join(os.getcwd(), "conversations", "faces")
        os.makedirs(faces_dir, exist_ok=True)
        
        print("Starting face detection using InsightFace...")
        print(f"Current recognition threshold: {self.recognition_threshold}")
        
        # Use our improved face detection method
        faces = self.app.get_with_multiple_sizes(frame)
        
        if not faces or len(faces) == 0:
            print("No face detected in the frame after trying multiple detection sizes")
            return None, False
        
        # Save a debug image showing all detected faces
        self._save_debug_face_image(frame, faces, "input")
        
        # Get the largest face in the frame (usually the most prominent)
        largest_face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
        face_embedding = largest_face.embedding
        
        print(f"Face detected with confidence: {largest_face.det_score:.4f}")
        
        # Create a temporary file path
        temp_dir = os.path.join(os.getcwd(), "temp_files")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"temp_face_{int(time.time())}.jpg")
        
        # Save the face to the temporary file with margin
        bbox = largest_face.bbox.astype(int)
        margin = 50  # Add margin around the face
        x1 = max(0, bbox[0] - margin)
        y1 = max(0, bbox[1] - margin)
        x2 = min(frame.shape[1], bbox[2] + margin)
        y2 = min(frame.shape[0], bbox[3] + margin)
        face_img = frame[y1:y2, x1:x2]
        cv2.imwrite(temp_file_path, face_img)
        
        print(f"Saved temporary face image to: {temp_file_path}")
        
        # Check if there are any existing faces
        existing_faces = [f for f in os.listdir(faces_dir) if f.startswith("face_") and f.endswith(".jpg")]
        
        if not existing_faces:
            print("No existing faces found in directory. Adding this face as the first one.")
            # No existing faces, add this as the first one
            face_id = f"face_{int(time.time())}"
            permanent_path = os.path.join(faces_dir, f"{face_id}.jpg")
            os.rename(temp_file_path, permanent_path)
            print(f"Nothing found adding temp file as: {face_id}")
            self.logger.info(f"Added first face with ID: {face_id}")
            return face_id, True
        
        print(f"Found {len(existing_faces)} existing faces. Checking for matches...")
        print(f"Using recognition threshold: {self.recognition_threshold}")
        
        # Compare with existing faces
        best_match = None
        best_similarity = 0
        all_similarities = []
        
        for face_file in existing_faces:
            face_path = os.path.join(faces_dir, face_file)
            try:
                print(f"Comparing with existing face: {face_file}")
                # Load the image and extract face embedding
                img = cv2.imread(face_path)
                if img is None:
                    print(f"Failed to load image: {face_path}")
                    continue
                    
                # Use our improved face detection with multiple sizes
                faces_in_img = self.app.get_with_multiple_sizes(img)
                
                if not faces_in_img or len(faces_in_img) == 0:
                    print(f"No face detected in {face_file}")
                    continue
                
                # Save debug image for this comparison
                self._save_debug_face_image(img, faces_in_img, f"compare_{face_file.split('.')[0]}")
                
                existing_embedding = faces_in_img[0].embedding
                # Calculate cosine similarity
                similarity = self._calculate_similarity(face_embedding, existing_embedding)
                all_similarities.append((face_file, similarity))
                
                print(f"Similarity with {face_file}: {similarity:.4f} (threshold: {self.recognition_threshold})")
                
                if similarity > self.recognition_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = face_file.split(".")[0]  # Remove extension
                    print(f"New best match: {best_match} with similarity: {best_similarity:.4f}")
            except Exception as e:
                        print(f"Error processing face file {face_file}: {e}")
        
        # Print summary of all similarities
        print("\nSIMILARITY SUMMARY:")
        for face_file, similarity in sorted(all_similarities, key=lambda x: x[1], reverse=True):
            match_indicator = "✓" if similarity > self.recognition_threshold else "✗"
            print(f"{match_indicator} {face_file}: {similarity:.4f}")
        print("")
        
        if best_match:
            # Found a match, delete the temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            print(f"Found matching face: {best_match} (similarity: {best_similarity:.4f})")
            print(f"Found matching face deleting temp and not doing anything")
            self.logger.info(f"Matched existing face: {best_match} (similarity: {best_similarity:.4f})")
            return best_match, False
        else:
            # No match found, add as a new face
            face_id = f"face_{int(time.time())}"
            permanent_path = os.path.join(faces_dir, f"{face_id}.jpg")
            os.rename(temp_file_path, permanent_path)
            print(f"No matching face found adding new face with ID: {face_id}")
            self.logger.info(f"Added new face with ID: {face_id}")
            return face_id, True
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two face embeddings
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Check for None values
            if embedding1 is None or embedding2 is None:
                print("Warning: Received None embedding in similarity calculation")
                return 0
                
            # Check for NaN values
            if np.isnan(embedding1).any() or np.isnan(embedding2).any():
                print("Warning: NaN values in embeddings")
                return 0
                
            # Basic cosine similarity calculation
            # Normalize vectors explicitly to ensure correct calculation
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                print("Warning: Zero norm detected in embeddings")
                return 0
                
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Ensure score is in range [0,1]
            similarity = float(max(0, min(1, similarity)))
            
            return similarity
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0

def merge_person_folders_cli():
    """Command-line interface to merge person folders."""
    print("This functionality has been deactivated since we no longer use Person directories.")
    return

if __name__ == "__main__":
    merge_person_folders_cli()