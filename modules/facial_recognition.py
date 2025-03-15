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
            recognition_threshold: Threshold for face recognition (0.0-1.0)
            face_db_path: Path to save face database (now only used for speaker mapping)
            model_name: InsightFace model name
        """
        # Set up logging
        self.logger = logging.getLogger("facial_recognition")
        
        # Set recognition threshold
        self.recognition_threshold = recognition_threshold
        
        # Initialize InsightFace
        try:
            self.app = FaceAnalysis(name=model_name)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.logger.info(f"Initialized InsightFace with model {model_name}")
        except Exception as e:
            self.logger.error(f"Error initializing InsightFace: {e}")
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
        Save conversation data for a specific person.
        Uses face images instead of pickle files for face data.
        
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
            
            # Ensure we're using a timestamp-based name
            if not person_name.startswith("Person_") and not person_name.startswith("Unknown_"):
                if not hasattr(self, '_session_timestamp'):
                    self._session_timestamp = int(time.time())
                person_name = f"Person_{self._session_timestamp}"
            
            # Create or get person directory
            person_dir = os.path.join(base_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            # If we have a face image in the conversation data, save it
            if 'face_image' in conversation_data and conversation_data['face_image'] is not None:
                self._save_face_image(person_name, conversation_data['face_image'])
            
            # Get the next conversation number
            conversation_files = [f for f in os.listdir(person_dir) 
                               if f.startswith("conversation_") and f.endswith(".txt") and not f == "conversation_history.txt"]
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
            
            # Update conversation history
            history_path = os.path.join(person_dir, "conversation_history.txt")
            history_exists = os.path.exists(history_path)
            
            with open(history_path, 'a' if history_exists else 'w') as f:
                if not history_exists:
                    f.write(f"CONVERSATION HISTORY FOR {person_name.upper()}\n")
                    f.write("="*80 + "\n\n")
                else:
                    f.write(f"\n\n--- NEW CONVERSATION {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")
                
                # Add the conversation content
                segments = conversation_data.get("segments", [])
                if segments:
                    for segment in segments:
                        speaker = segment.get("speaker", "Unknown")
                        person = segment.get("person", speaker)
                        text = segment.get("text", "")
                        f.write(f"[{person}]: {text}\n")
                elif "text" in conversation_data:
                    f.write(conversation_data.get("text", ""))
            
            # Extend knowledge base if provided
            if "knowledge_base" in conversation_data and conversation_data["knowledge_base"]:
                kb_path = os.path.join(person_dir, "knowledge_base.txt")
                kb_exists = os.path.exists(kb_path)
                
                with open(kb_path, 'a' if kb_exists else 'w') as f:
                    if not kb_exists:
                        f.write(f"KNOWLEDGE BASE FOR {person_name.upper()}\n")
                        f.write("="*80 + "\n\n")
                    else:
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
                        topics_exists = os.path.exists(topics_path)
                        
                        with open(topics_path, 'a' if topics_exists else 'w') as f:
                            if not topics_exists:
                                f.write(f"TOPICS FOR {person_name.upper()}\n")
                                f.write("="*80 + "\n\n")
                            else:
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
        Merge two person folders, moving all content from source to target.
        Also updates the identity mappings.
        
        Args:
            source_id: Source person ID (e.g., "Person_1742069504")
            target_id: Target person ID (e.g., "Person_1742069534")
            
        Returns:
            bool: True if merged successfully
        """
        try:
            base_dir = os.path.join(os.getcwd(), "conversations")
            source_dir = os.path.join(base_dir, source_id)
            target_dir = os.path.join(base_dir, target_id)
            
            # Check if both directories exist
            if not os.path.exists(source_dir):
                self.logger.error(f"Source directory {source_dir} does not exist")
                return False
            
            if not os.path.exists(target_dir):
                self.logger.error(f"Target directory {target_dir} does not exist")
                return False
            
            # Create identity mapping
            if not hasattr(self, 'identity_mappings'):
                self.identity_mappings = {}
            
            # Add mapping from source to target
            self.identity_mappings[source_id] = target_id
            self._save_identity_mappings()
            
            # Move all files from source to target
            for item in os.listdir(source_dir):
                source_path = os.path.join(source_dir, item)
                target_path = os.path.join(target_dir, item)
                
                # Handle conversation files specially
                if item.startswith("conversation_") and item.endswith(".txt"):
                    # Find the next conversation number in target
                    conv_files = [f for f in os.listdir(target_dir) 
                               if f.startswith("conversation_") and f.endswith(".txt") 
                               and not f == "conversation_history.txt"]
                    next_number = 1
                    if conv_files:
                        numbers = [int(f.split("_")[1].split(".")[0]) for f in conv_files]
                        next_number = max(numbers) + 1
                    
                    # Copy with new number
                    new_target_path = os.path.join(target_dir, f"conversation_{next_number}.txt")
                    with open(source_path, 'r') as src, open(new_target_path, 'w') as dst:
                        content = src.read()
                        # Replace source ID with target ID
                        content = content.replace(source_id.upper(), target_id.upper())
                        dst.write(content)
                    
                    # Also append to conversation history
                    history_path = os.path.join(target_dir, "conversation_history.txt")
                    with open(history_path, 'a') as f:
                        f.write(f"\n\n--- MERGED FROM {source_id} ---\n\n")
                        with open(source_path, 'r') as src:
                            lines = src.readlines()
                            if len(lines) > 3:  # Skip title and separator
                                f.writelines(lines[3:])
                
                # Handle face images
                elif item.startswith("face_") and item.endswith(".jpg"):
                    # Copy face images with original names
                    import shutil
                    shutil.copy2(source_path, target_path)
                
                # Handle other files
                elif item not in ["face.jpg"]:  # Skip main face.jpg as target has its own
                    # For other files like knowledge_base.txt, merge content
                    if os.path.exists(target_path) and item.endswith(".txt"):
                        with open(target_path, 'a') as f:
                            f.write(f"\n\n--- MERGED FROM {source_id} ---\n\n")
                            with open(source_path, 'r') as src:
                                f.write(src.read())
                    else:
                        # Just copy the file
                        import shutil
                        shutil.copy2(source_path, target_path)
            
            # Update known_faces to point to target_id
            if source_id in self.known_faces:
                if target_id not in self.known_faces:
                    self.known_faces[target_id] = self.known_faces[source_id]
                del self.known_faces[source_id]
                self._save_face_db()
            
            # Update face galleries
            if hasattr(self, 'face_galleries') and source_id in self.face_galleries:
                if target_id not in self.face_galleries:
                    self.face_galleries[target_id] = []
                self.face_galleries[target_id].extend(self.face_galleries[source_id])
                del self.face_galleries[source_id]
            
            self.logger.info(f"Successfully merged {source_id} into {target_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error merging person folders: {e}")
            import traceback
            traceback.print_exc()
            return False

    def find_existing_person_folder(self, face_image: np.ndarray) -> str:
        """
        Try to find an existing person folder that matches the current face.
        
        Args:
            face_image: Current face image
            
        Returns:
            str: Person folder name if found, None otherwise
        """
        try:
            # Detect and get face embedding
            faces = self.app.get(face_image)
            if not faces:
                return None
            
            # Use the largest face
            if len(faces) > 1:
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            
            face_embedding = faces[0].normed_embedding
            
            # Try to match against all known faces with a lower threshold
            best_match = None
            best_similarity = 0.0
            
            # First check against all known faces
            for name, known_embedding in self.known_faces.items():
                similarity = np.dot(face_embedding, known_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
            
            # Then check against all face galleries
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
            
            # Use a lower threshold for finding existing folders (0.08 instead of 0.15)
            if best_similarity >= 0.08 and best_match:
                self.logger.info(f"Found existing person folder: {best_match} with similarity {best_similarity:.2f}")
                return best_match
            
            return None
        except Exception as e:
            self.logger.error(f"Error finding existing person folder: {e}")
            return None

def merge_person_folders_cli():
    """Command-line interface to merge person folders."""
    import argparse
    from modules.facial_recognition import FacialRecognition
    
    parser = argparse.ArgumentParser(description='Merge person folders')
    parser.add_argument('source', help='Source person ID (e.g., Person_1742069504)')
    parser.add_argument('target', help='Target person ID (e.g., Person_1742069534)')
    args = parser.parse_args()
    
    # Initialize facial recognition
    fr = FacialRecognition()
    
    # Merge folders
    success = fr.merge_person_folders(args.source, args.target)
    
    if success:
        print(f"Successfully merged {args.source} into {args.target}")
    else:
        print(f"Failed to merge {args.source} into {args.target}")

if __name__ == "__main__":
    merge_person_folders_cli()