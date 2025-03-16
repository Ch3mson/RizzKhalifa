#!/usr/bin/env python3

import os
import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Set
import pickle
import logging
from pathlib import Path

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
            recognition_threshold: Threshold for face recognition similarity
            face_db_path: Path to save/load face embeddings database
            model_name: InsightFace model to use
        """
        self.recognition_threshold = recognition_threshold
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("FacialRecognition")
        
        # Initialize InsightFace
        try:
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'], 
                                   allowed_modules=['detection', 'recognition'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.logger.info(f"InsightFace initialized with model: {model_name}")
        except Exception as e:
            self.logger.error(f"Error initializing InsightFace: {e}")
            print(f"Error initializing InsightFace: {e}")
            raise
            
        # Database for storing face embeddings - use user's current project directory
        project_dir = os.getcwd()
        if not face_db_path:
            face_db_dir = os.path.join(project_dir, "conversations")
            os.makedirs(face_db_dir, exist_ok=True)
            face_db_path = os.path.join(face_db_dir, "face_db.pkl")
        
        self.face_db_path = face_db_path
        self.known_faces = {}  # name -> embedding
        self.speaker_face_mapping = {}  # speaker_id -> person_name
        self.face_count = 0  # Counter for auto-naming unknown faces
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.face_db_path), exist_ok=True)
        
        # Load existing face database if it exists
        self._load_face_db()
        
        # Track the current face for this conversation
        self.current_face_name = None
        self.current_face_embedding = None
        self.last_face_check_time = 0
    
    def _load_face_db(self) -> None:
        """Load the face embeddings database from disk if it exists."""
        if os.path.exists(self.face_db_path):
            try:
                with open(self.face_db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data.get('faces', {})
                    self.speaker_face_mapping = data.get('mapping', {})
                    # Get the counter for auto-naming from the database, or use default
                    self.face_count = data.get('face_count', 0)
                self.logger.info(f"Loaded {len(self.known_faces)} faces from database at {self.face_db_path}")
            except Exception as e:
                self.logger.error(f"Error loading face database: {e}")
                self.known_faces = {}
                self.speaker_face_mapping = {}
    
    def _save_face_db(self) -> None:
        """Save the face embeddings database to disk."""
        try:
            with open(self.face_db_path, 'wb') as f:
                pickle.dump({
                    'faces': self.known_faces,
                    'mapping': self.speaker_face_mapping,
                    'face_count': self.face_count
                }, f)
            self.logger.info(f"Saved {len(self.known_faces)} faces to database at {self.face_db_path}")
        except Exception as e:
            self.logger.error(f"Error saving face database: {e}")
            
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
        """Return the interval for rechecking faces"""
        return 30  # seconds
    
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
    
    def process_video_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process a video frame to detect and recognize faces.
        
        Args:
            frame: Video frame as numpy array
            
        Returns:
            List[Dict]: List of detected faces with recognition results
        """
        results = []
        
        try:
            # Get faces with bounding boxes and embeddings
            faces = self.app.get(frame)
            
            for i, face in enumerate(faces):
                # Extract face information
                bbox = face.bbox.astype(int)
                name, confidence = "unknown", 0.0
                
                # Try to recognize using the embedding
                face_embedding = face.normed_embedding
                
                # Find the closest match
                best_match = "unknown"
                best_similarity = 0.0
                
                for person_name, known_embedding in self.known_faces.items():
                    # Calculate cosine similarity
                    similarity = np.dot(face_embedding, known_embedding)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = person_name
                
                # Check if similarity exceeds threshold
                if best_similarity >= self.recognition_threshold:
                    name, confidence = best_match, float(best_similarity)
                else:
                    # Automatically add new face with a generated name
                    self.face_count += 1
                    new_name = f"Person_{self.face_count}"
                    self.known_faces[new_name] = face_embedding
                    self._save_face_db()
                    name, confidence = new_name, 1.0  # New face is 100% match to itself
                    self.logger.info(f"Added new face automatically: {new_name}")
                
                # Add result
                results.append({
                    "id": i,
                    "bbox": bbox.tolist(),
                    "name": name,
                    "confidence": confidence,
                    "embedding": face_embedding
                })
                
        except Exception as e:
            self.logger.error(f"Error processing video frame: {e}")
        
        return results
    
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
                                       knowledge_base: Dict = None) -> List[Dict]:
        """
        Process a conversation with video, associating speakers with faces.
        Simplified to capture just one face per conversation and wait until one is found.
        
        Args:
            video_file: Path to video file
            diarized_segments: List of diarized speech segments with speaker labels
            output_dir: Directory to save conversation data
            knowledge_base: Optional knowledge base to save with the conversation
            
        Returns:
            List[Dict]: Updated diarized segments with person names
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                self.logger.error(f"Could not open video file: {video_file}")
                return diarized_segments
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize variables for face detection
            face_detected = False
            detected_person = None
            max_attempts = 10  # Maximum number of attempts to find a face
            attempts = 0
            
            # Keep trying until we find a face or reach max attempts
            while not face_detected and attempts < max_attempts:
                attempts += 1
                
                # Get a frame from the middle of the video
                middle_frame = max(0, total_frames // 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                ret, frame = cap.read()
                
                if not ret:
                    self.logger.warning(f"Failed to read frame (attempt {attempts}/{max_attempts})")
                    continue
                
                # Process the frame to detect faces
                face_results = self.process_video_frame(frame)
                
                if face_results:
                    # We found at least one face
                    face_detected = True
                    # Use the first detected face
                    face_info = face_results[0]
                    detected_person = face_info["name"]
                    self.logger.info(f"Detected person: {detected_person}")
                    
                    # Create a person directory
                    person_dir = os.path.join(output_dir, detected_person)
                    os.makedirs(person_dir, exist_ok=True)
                    
                    # If it's a new person, save a reference image
                    ref_image_dir = os.path.join(person_dir, "reference_images")
                    os.makedirs(ref_image_dir, exist_ok=True)
                    ref_image_path = os.path.join(ref_image_dir, f"{detected_person}_{int(time.time())}.jpg")
                    cv2.imwrite(ref_image_path, frame)
                    
                    # Associate all speaker segments with this person
                    for segment in diarized_segments:
                        segment["person"] = detected_person
                    
                    # Save the full conversation data
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    conversation_data = {
                        "timestamp": timestamp,
                        "segments": diarized_segments,
                        "knowledge_base": knowledge_base or {}
                    }
                    
                    self.save_conversation_for_person(
                        person_name=detected_person,
                        conversation_data=conversation_data,
                        base_dir=output_dir
                    )
                else:
                    # Try another frame if no face detected
                    random_frame = int(np.random.uniform(0, total_frames))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
            
            # If no face was detected after multiple attempts
            if not face_detected:
                self.logger.warning("No face detected in video after multiple attempts")
                # Create a unique identifier with timestamp to ensure uniqueness
                unknown_id = f"Unknown_Person_{int(time.time())}"
                self.logger.info(f"Created unique unknown person ID: {unknown_id}")
                
                # Create an unknown person entry
                for segment in diarized_segments:
                    segment["person"] = unknown_id
                
                # Save the conversation data anyway under an unknown person ID
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                conversation_data = {
                    "timestamp": timestamp,
                    "segments": diarized_segments,
                    "knowledge_base": knowledge_base or {}
                }
                
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
        Save conversation data for a specific person.
        
        Args:
            person_name: Name of the person
            conversation_data: Conversation data to save (including knowledge base)
            base_dir: Base directory to save conversations
            
        Returns:
            bool: True if saved successfully
        """
        try:
            # Ensure person_name is a unique identifier by adding a timestamp if it's a generic name
            if person_name.startswith("Person_") or person_name.startswith("Unknown"):
                # Check if this person already has a folder before adding timestamp
                existing_folder = self.find_existing_person_folder(person_name, base_dir or os.path.join(os.getcwd(), "conversations"))
                if existing_folder:
                    self.logger.info(f"Using existing folder for {person_name}: {existing_folder}")
                    person_name = existing_folder
                else:
                    # Only add timestamp if no existing folder found
                    unique_id = f"{person_name}_{int(time.time())}"
                    self.logger.info(f"Converting generic name {person_name} to unique ID: {unique_id}")
                    person_name = unique_id
            
            # Set up directory in the current project folder
            base_dir = base_dir or os.path.join(os.getcwd(), "conversations")
            person_dir = os.path.join(base_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            # Save face embedding if available
            if self.current_face_embedding is not None:
                self.save_face_embedding(person_name, self.current_face_embedding, base_dir)
                self.logger.info(f"Saved face embedding for {person_name}")
            
            self.logger.info(f"Saving conversation for {person_name} in project directory: {person_dir}")
            
            # Create conversation directory with timestamp
            timestamp = conversation_data.get("timestamp") or time.strftime("%Y%m%d-%H%M%S")
            conversation_dir = os.path.join(person_dir, f"conversation_{timestamp}")
            os.makedirs(conversation_dir, exist_ok=True)
            
            # Save conversation data
            conversation_path = os.path.join(conversation_dir, "conversation_data.pkl")
            with open(conversation_path, 'wb') as f:
                pickle.dump(conversation_data, f)
            
            # Extract key information
            segments = conversation_data.get("segments", [])
            knowledge_base = conversation_data.get("knowledge_base", {})
            
            # Save knowledge base separately for easier access
            if knowledge_base:
                kb_path = os.path.join(conversation_dir, "knowledge_base.pkl")
                with open(kb_path, 'wb') as f:
                    pickle.dump(knowledge_base, f)
                
                # Also save a text version of the knowledge base for human reading
                kb_text_path = os.path.join(conversation_dir, "knowledge_base.txt")
                with open(kb_text_path, 'w') as f:
                    f.write(f"Knowledge Base for {person_name} - {timestamp}\n")
                    f.write("="*80 + "\n\n")
                    
                    for topic, snippets in knowledge_base.items():
                        f.write(f"TOPIC: {topic}\n")
                        f.write("-"*80 + "\n")
                        for i, snippet in enumerate(snippets):
                            f.write(f"{i+1}. {snippet}\n\n")
                        f.write("\n")
            
            # Save a text version of the conversation
            text_path = os.path.join(conversation_dir, "conversation.txt")
            with open(text_path, 'w') as f:
                f.write(f"Conversation with {person_name} on {timestamp}\n")
                f.write("="*80 + "\n\n")
                
                # Get full text from segments for easy reading
                full_text = ""
                if segments:
                    for segment in segments:
                        speaker = segment.get("speaker", "Unknown")
                        person = segment.get("person", speaker)
                        text = segment.get("text", "")
                        start = segment.get("start", 0)
                        end = segment.get("end", 0)
                        full_text += f"{text} "
                        f.write(f"[{person}] ({start:.2f}s - {end:.2f}s): {text}\n")
                        
                f.write("\n\nFULL TEXT:\n")
                f.write("-"*80 + "\n")
                f.write(full_text.strip())
            
            # Create a comprehensive summary file that includes all information
            summary_path = os.path.join(conversation_dir, "summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"CONVERSATION SUMMARY FOR {person_name.upper()} - {timestamp}\n")
                f.write("="*80 + "\n\n")
                
                # Add full conversation
                if segments:
                    f.write("FULL CONVERSATION:\n")
                    f.write("-"*80 + "\n")
                    for segment in segments:
                        speaker = segment.get("speaker", "Unknown")
                        person = segment.get("person", speaker)
                        text = segment.get("text", "")
                        f.write(f"[{person}]: {text}\n")
                    f.write("\n\n")
                
                # Add knowledge base summary
                if knowledge_base:
                    f.write("KNOWLEDGE BASE TOPICS:\n")
                    f.write("-"*80 + "\n")
                    for topic, snippets in knowledge_base.items():
                        f.write(f"• {topic}: {len(snippets)} snippets\n")
                        # Add the first snippet as a preview
                        if snippets:
                            preview = snippets[0][:150] + "..." if len(snippets[0]) > 150 else snippets[0]
                            f.write(f"  Preview: {preview}\n\n")
                    f.write("\n")
                    
                # Add file index
                f.write("FILES AVAILABLE IN THIS CONVERSATION:\n")
                f.write("-"*80 + "\n")
                f.write(f"• conversation_data.pkl - Complete data in machine-readable format\n")
                f.write(f"• conversation.txt - Human-readable conversation transcript\n")
                if knowledge_base:
                    f.write(f"• knowledge_base.pkl - Machine-readable knowledge information\n")
                    f.write(f"• knowledge_base.txt - Human-readable knowledge information\n")
                f.write(f"• summary.txt - This summary file\n\n")
                
                # Add timestamp and location information
                f.write("METADATA:\n")
                f.write("-"*80 + "\n")
                f.write(f"• Timestamp: {timestamp}\n")
                f.write(f"• Person ID: {person_name}\n")
                f.write(f"• Storage Location: {conversation_dir}\n")
            
            self.logger.info(f"Saved comprehensive conversation data for {person_name} in {conversation_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving conversation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
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

    def update_current_face(self, face_name: str, face_embedding: np.ndarray) -> None:
        """
        Update the current face for this conversation session.
        
        Args:
            face_name: Name of the person
            face_embedding: Face embedding vector
        """
        self.current_face_name = face_name
        self.current_face_embedding = face_embedding
        self.last_face_check_time = time.time()
        
        # Also make sure it's saved in the known faces
        if face_name not in self.known_faces:
            self.known_faces[face_name] = face_embedding
            self._save_face_db()
            self.logger.info(f"Added new face '{face_name}' to known faces database")

    def is_same_face(self, face_embedding: np.ndarray) -> bool:
        """
        Check if a face embedding corresponds to the current face.
        
        Args:
            face_embedding: Face embedding to check
            
        Returns:
            bool: True if it's the same face, False otherwise
        """
        if self.current_face_embedding is None:
            return False
            
        similarity = np.dot(face_embedding, self.current_face_embedding)
        return similarity >= self.recognition_threshold 

    def find_existing_person_folder(self, person_name, base_dir):
        """
        Find an existing folder for a recognized person.
        
        Args:
            person_name: The name of the recognized person
            base_dir: The base directory to search in
            
        Returns:
            str: Path to existing folder or None if not found
        """
        try:
            # Skip if the base directory doesn't exist
            if not os.path.exists(base_dir):
                return None
            
            # First, check for exact match
            exact_path = os.path.join(base_dir, person_name)
            if os.path.exists(exact_path) and os.path.isdir(exact_path):
                self.logger.info(f"Found exact matching folder: {exact_path}")
                return person_name
            
            # If no exact match, check for folders with the same base name
            # (e.g., "Person_123" should match "Person_123_xyz")
            if "_" in person_name:
                base_name = person_name.split("_")[0]  # Get "Person" from "Person_123"
                
                # Look for directories that start with this base name
                matching_dirs = []
                for item in os.listdir(base_dir):
                    item_path = os.path.join(base_dir, item)
                    if os.path.isdir(item_path) and item.startswith(base_name):
                        # Check if this directory has conversation files
                        if (os.path.exists(os.path.join(item_path, "conversation_history.txt")) or 
                            os.path.exists(os.path.join(item_path, "conversation.txt")) or
                            os.path.exists(os.path.join(item_path, "conversation_1.txt"))):
                            matching_dirs.append(item)
                
                if matching_dirs:
                    # If we found matching directories, check if any of them match our face
                    for dir_name in matching_dirs:
                        # Try to load face embedding from this directory
                        embedding_path = os.path.join(base_dir, dir_name, "face_embedding.npy")
                        if os.path.exists(embedding_path):
                            try:
                                stored_embedding = np.load(embedding_path)
                                
                                # Compare with current face embedding
                                if self.current_face_embedding is not None:
                                    similarity = np.dot(self.current_face_embedding, stored_embedding)
                                    self.logger.info(f"Comparing face with folder {dir_name}: similarity {similarity:.2f}")
                                    if similarity >= self.recognition_threshold:
                                        self.logger.info(f"Face match found in folder {dir_name} with similarity {similarity:.2f}")
                                        return dir_name
                            except Exception as e:
                                self.logger.error(f"Error comparing face embeddings: {e}")
                    
                    # If no face embedding match, return the most recent directory
                    if self.current_face_embedding is None:
                        self.logger.info(f"No face embedding to compare, using most recent directory: {matching_dirs[-1]}")
                        return matching_dirs[-1]
        
            return None
        except Exception as e:
            self.logger.error(f"Error finding existing person folder: {e}")
            return None

    def save_face_embedding(self, person_name, embedding, base_dir):
        """Save face embedding to the person's folder"""
        try:
            person_dir = os.path.join(base_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            # Save embedding as numpy file
            embedding_path = os.path.join(person_dir, "face_embedding.npy")
            np.save(embedding_path, embedding)
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving face embedding: {e}")
            return False 