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

class FacialRecognitionModule:
    """
    Module for facial recognition that works alongside speaker diarization.
    
    This module uses InsightFace to recognize faces in video frames and 
    associates them with speakers identified by the diarization system.
    """
    
    def __init__(self, 
                recognition_threshold: float = 0.4,
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
            raise
            
        # Database for storing face embeddings - use user's current project directory
        project_dir = os.getcwd()
        if not face_db_path:
            # Create a separate system directory for face database
            system_dir = os.path.join(project_dir, "conversations", "system_data")
            os.makedirs(system_dir, exist_ok=True)
            face_db_path = os.path.join(system_dir, "face_db.pkl")
        
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
        """Get the current face recheck interval in seconds."""
        global FACE_RECHECK_INTERVAL
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
                    # Ensure this name is stable by using just one counter
                    self.face_count += 1
                    
                    # Create a consistent name with just one timestamp
                    if not hasattr(self, '_session_timestamp'):
                        self._session_timestamp = int(time.time())
                    
                    new_name = f"Person_{self._session_timestamp}"
                    # Only increment if we're creating different faces
                    if len(self.known_faces) > 0:
                        new_name = f"Person_{self._session_timestamp}_{self.face_count}"
                        
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
                                       knowledge_base: Dict = None,
                                       workflow_state: Dict = None) -> List[Dict]:
        """
        Process a conversation with video, associating speakers with faces.
        Simplified to capture just one face per conversation and wait until one is found.
        
        Args:
            video_file: Path to video file
            diarized_segments: List of diarized speech segments with speaker labels
            output_dir: Directory to save conversation data
            knowledge_base: Optional knowledge base to save with the conversation
            workflow_state: Optional full workflow state for additional context
            
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
                    
                    # Clean up person name for consistency
                    if detected_person.startswith("Person_") or detected_person.startswith("Unknown"):
                        parts = detected_person.split("_")
                        if len(parts) > 1:
                            # Use just the base name + first identifier
                            detected_person = f"{parts[0]}_{parts[1]}"
                            self.logger.info(f"Using cleaned name {detected_person} for consistency")
                    
                    # Create a person directory
                    person_dir = os.path.join(output_dir, detected_person)
                    os.makedirs(person_dir, exist_ok=True)
                    
                    # If it's a new person, save a reference image
                    ref_image_dir = os.path.join(person_dir, "reference_images")
                    os.makedirs(ref_image_dir, exist_ok=True)
                    ref_image_path = os.path.join(ref_image_dir, f"reference_{int(time.time())}.jpg")
                    cv2.imwrite(ref_image_path, frame)
                    
                    # Associate all speaker segments with this person
                    for segment in diarized_segments:
                        segment["person"] = detected_person
                    
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
                # Create a unique identifier that will remain consistent
                current_time = int(time.time())
                unknown_id = f"Person_{current_time}"
                self.logger.info(f"Created unique person ID: {unknown_id}")
                
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
        Save conversation data for a specific person.
        If the person is new, saves their face embedding. If unknown, tries to match with existing faces.
        
        Args:
            person_name: Name of the person
            conversation_data: Conversation data to save (including knowledge base)
            base_dir: Base directory to save conversations
            
        Returns:
            bool: True if saved successfully
        """
        try:
            # Set up directory in the current project folder
            base_dir = base_dir or os.path.join(os.getcwd(), "conversations")
            
            # If this is an unknown person, try to match with existing faces
            if person_name.startswith("Person_") or person_name.startswith("Unknown"):
                # Get the current face embedding if we have one
                current_embedding = None
                if hasattr(self, 'current_face_embedding') and self.current_face_embedding is not None:
                    current_embedding = self.current_face_embedding
                
                # Try to match with existing faces
                if current_embedding is not None:
                    best_match = None
                    best_similarity = 0.0
                    
                    # Look through all person directories
                    for dir_name in os.listdir(base_dir):
                        dir_path = os.path.join(base_dir, dir_name)
                        if os.path.isdir(dir_path) and (dir_name.startswith("Person_") or dir_name.startswith("Unknown")):
                            # Check for face embedding file
                            face_file = os.path.join(dir_path, "face_embedding.pkl")
                            if os.path.exists(face_file):
                                try:
                                    with open(face_file, 'rb') as f:
                                        saved_embedding = pickle.load(f)
                                    similarity = float(np.dot(current_embedding, saved_embedding))
                                    if similarity > best_similarity and similarity >= self.recognition_threshold:
                                        best_similarity = similarity
                                        best_match = dir_name
                                except Exception as e:
                                    self.logger.error(f"Error loading face embedding from {face_file}: {e}")
                    
                    if best_match:
                        self.logger.info(f"Matched face to existing person {best_match} with similarity {best_similarity:.4f}")
                        person_name = best_match
            
            # Create or get person directory
            person_dir = os.path.join(base_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            # Save face embedding if we have one
            if hasattr(self, 'current_face_embedding') and self.current_face_embedding is not None:
                face_file = os.path.join(person_dir, "face_embedding.pkl")
                with open(face_file, 'wb') as f:
                    pickle.dump(self.current_face_embedding, f)
                self.logger.info(f"Saved face embedding for {person_name}")
            
            # Get the next conversation number
            conversation_files = [f for f in os.listdir(person_dir) 
                               if f.startswith("conversation_") and f.endswith(".txt")]
            next_number = 1
            if conversation_files:
                numbers = [int(f.split("_")[1].split(".")[0]) for f in conversation_files]
                next_number = max(numbers) + 1
            
            # Save new conversation with incremented number
            conversation_path = os.path.join(person_dir, f"conversation_{next_number}.txt")
            with open(conversation_path, 'w') as f:
                f.write(f"CONVERSATION {next_number} WITH {person_name.upper()}\n")
                f.write("="*80 + "\n\n")
                
                # Get full text from segments for easy reading
                segments = conversation_data.get("segments", [])
                if segments:
                    for segment in segments:
                        speaker = segment.get("speaker", "Unknown")
                        person = segment.get("person", speaker)
                        text = segment.get("text", "")
                        f.write(f"[{person}]: {text}\n")
                elif "text" in conversation_data:
                    # If we have just raw text without segments
                    f.write(conversation_data.get("text", ""))
            
            # Extend knowledge base if provided
            if "knowledge_base" in conversation_data and conversation_data["knowledge_base"]:
                kb_path = os.path.join(person_dir, "knowledge_base.txt")
                
                with open(kb_path, 'a') as f:
                    f.write(f"\n\n--- UPDATED KNOWLEDGE BASE {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")
                    
                    for topic, snippets in conversation_data["knowledge_base"].items():
                        f.write(f"TOPIC: {topic}\n")
                        f.write("-"*80 + "\n")
                        for i, snippet in enumerate(snippets):
                            f.write(f"{i+1}. {snippet}\n\n")
                        f.write("\n")
            
            # Extend topics if available
            if "workflow_state" in conversation_data:
                state = conversation_data.get("workflow_state", {})
                if "topics" in state:
                    topics = state.get("topics", [])
                    if topics:
                        topics_path = os.path.join(person_dir, "topics.txt")
                        
                        with open(topics_path, 'a') as f:
                            f.write(f"\n\n--- UPDATED TOPICS {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")
                            
                            # Organize topics by category
                            categorized_topics = {
                                "Likes": [],
                                "Dislikes": [],
                                "School": [],
                                "Hobbies": [],
                                "Professional": [],
                                "Skills": []
                            }
                            
                            # Sort topics into categories
                            for topic in topics:
                                if isinstance(topic, dict):
                                    category = topic.get("category", "")
                                    if category in categorized_topics:
                                        categorized_topics[category].append({
                                            "name": topic.get("name", ""),
                                            "description": topic.get("description", "")
                                        })
                            
                            # Write topics by category
                            for category, items in categorized_topics.items():
                                if items:  # Only write categories that have items
                                    f.write(f"{category.upper()}:\n")
                                    f.write("-"*40 + "\n")
                                    for item in items:
                                        f.write(f"• {item['name']}: {item['description']}\n")
                                    f.write("\n")
            
            self.logger.info(f"Successfully saved data for {person_name} with incremental conversation numbering")
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