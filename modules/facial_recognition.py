#!/usr/bin/env python3

import os
import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Set
import logging
import json

from insightface.app import FaceAnalysis

from modules.config import SAMPLE_RATE

class ImprovedFaceAnalysis(FaceAnalysis):
    """
    Extension of InsightFace's FaceAnalysis that offers improved face detection capabilities.
    This class provides better handling of detection sizes and implements adaptive detection.
    """
    
    def get_with_multiple_sizes(self, img, max_num=0, sizes=None):
        """
        Attempts to detect faces using multiple detection sizes.
        """
        if sizes is None:
            sizes = [(640, 640), (320, 320), (480, 480), (720, 720), (960, 960)]
        
        print(f"Trying to detect faces with multiple detection sizes: {sizes}")
        faces = None
        
        for det_size in sizes:
            try:
                if hasattr(self.det_model, "input_size"):
                    self.det_model.input_size = det_size
                
                faces = self.get(img, max_num)
                if faces and len(faces) > 0:
                    print(f"Successfully detected {len(faces)} faces with detection size {det_size}")
                    return faces
            except Exception as e:
                print(f"Error with detection size {det_size}: {e}")
                continue
        
        if not faces or len(faces) == 0:
            print("No faces detected with any detection size")
            return []
        
        return faces

class FacialRecognitionModule:
    """
    Module for facial recognition that works alongside speaker diarization.
    """
    
    def __init__(self, 
                recognition_threshold: float = 0.5, 
                face_db_path: str = None,
                model_name: str = 'buffalo_l'):
        self.logger = logging.getLogger("facial_recognition")
        
        self.recognition_threshold = recognition_threshold
        
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.app = ImprovedFaceAnalysis(name=model_name, providers=providers)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.logger.info(f"Initialized InsightFace with model {model_name}")
        except Exception as e:
            self.logger.error(f"Error initializing InsightFace: {e}")
            print(f"Error initializing InsightFace: {e}")
            raise
        
        project_dir = os.getcwd()
        
        if not face_db_path:
            system_dir = os.path.join(project_dir, "conversations", "system_data")
            os.makedirs(system_dir, exist_ok=True)
            face_db_path = os.path.join(system_dir, "speaker_mapping.json")
        
        self.face_db_path = face_db_path
        self.known_faces = {} 
        self.speaker_face_mapping = {} 
        self.face_count = 0  
        
        os.makedirs(os.path.dirname(self.face_db_path), exist_ok=True)
        
        self.current_face_name = None
        self.current_face_embedding = None
        self.last_face_check_time = 0
        
        self.clear_current_user_id()
        
        self._load_face_db()
        
        self._load_face_galleries()
        
        self._load_persistent_identities()
        
        self.identity_mappings = {}
        self._load_identity_mappings()
    
    def _load_face_db(self) -> None:
        """
        Load the face database by scanning conversation directories for face images.
        Also loads speaker mapping from JSON if available.
        """
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
        
        try:
            base_dir = os.path.join(os.getcwd(), "conversations")
            if not os.path.exists(base_dir):
                self.logger.info("No conversations directory found. Creating new database.")
                os.makedirs(base_dir, exist_ok=True)
                return
            
            person_count = 0
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                face_path = os.path.join(item_path, "face.jpg")
                
                if os.path.isdir(item_path) and os.path.exists(face_path):
                    try:
                        face_image = cv2.imread(face_path)
                        if face_image is None:
                            self.logger.warning(f"Could not read face image for {item}")
                            continue
                        
                        faces = self.app.get(face_image)
                        if not faces:
                            self.logger.warning(f"No face detected in saved image for {item}")
                            continue
                        
                        if len(faces) > 1:
                            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                        
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
            system_dir = os.path.join(os.getcwd(), "conversations", "system_data")
            os.makedirs(system_dir, exist_ok=True)
            
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
        """
        global FACE_RECHECK_INTERVAL
        FACE_RECHECK_INTERVAL = seconds
        self.logger.info(f"Face recheck interval set to {FACE_RECHECK_INTERVAL} seconds")
        
    def get_recheck_interval(self) -> int:
        """
        Get the interval for rechecking faces in seconds.
        """
        return FACE_RECHECK_INTERVAL
    
    def add_face(self, name: str, face_image: np.ndarray) -> bool:
        """
        Add a new face to the database with a name.
        """
        try:
            faces = self.app.get(face_image)
            
            if not faces:
                self.logger.warning(f"No face detected in the provided image for {name}")
                return False
            
            if len(faces) > 1:
                self.logger.info(f"Multiple faces detected, using the largest one for {name}")
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            
            face_embedding = faces[0].normed_embedding
            self.known_faces[name] = face_embedding
            
            self._save_face_db()
            self.logger.info(f"Added face for {name} to database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding face: {e}")
            return False
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Recognize a face in an image.
        """
        try:
            faces = self.app.get(face_image)
            
            if not faces:
                return "unknown", 0.0
            
            if len(faces) > 1:
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            
            face_embedding = faces[0].normed_embedding
            
            best_match = "unknown"
            best_similarity = 0.0
            
            for name, known_embedding in self.known_faces.items():
                similarity = np.dot(face_embedding, known_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
            
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
        """
        try:
            faces = self.app.get(frame)
            
            results = []
            
            if not faces:
                self.logger.debug("No faces detected in frame")
                return results
            
            for i, face in enumerate(faces):
                try:
                    face_embedding = face.embedding
                    bbox = face.bbox
                    
                    x1, y1, x2, y2 = [int(b) for b in bbox]
                    margin = 20
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame.shape[1], x2 + margin)
                    y2 = min(frame.shape[0], y2 + margin)
                    face_image = frame[y1:y2, x1:x2]
                    
                    name = None
                    confidence = 0.0
                    
                    if self.known_faces:
                        name = self.get_persistent_identity(face_embedding)
                        
                        if hasattr(self, 'identity_mappings') and name in self.identity_mappings:
                            canonical_name = self.identity_mappings[name]
                            self.logger.info(f"Mapped {name} to canonical identity: {canonical_name}")
                            name = canonical_name
                        
                        if name != "unknown":
                            confidence = 1.0 
                            self.logger.info(f"Recognized persistent identity: {name}")
                            
                            self.known_faces[name] = face_embedding
                            
                            self.add_to_face_gallery(name, face_embedding)
                            
                            self._save_face_db()
                            
                            self.update_current_face(name, face_embedding, face_image)
                            
                            self.write_current_user_id(name, is_new_face=False)
                    
                    if not name:
                        existing_person = self.find_existing_person_folder(face_image)
                        if existing_person:
                            person_id = existing_person
                            self.logger.info(f"Using existing person folder: {person_id}")
                        else:
                            current_time = int(time.time())
                            person_id = f"Person_{current_time}"
                            self.logger.info(f"Created new person folder: {person_id}")
                        
                        current_time = int(time.time())
                        new_name = f"Person_{current_time}"
                        if len(self.known_faces) > 0:
                            self.face_count += 1
                            new_name = f"Person_{current_time}_{self.face_count}"
                            
                        self.known_faces[new_name] = face_embedding
                        self._save_face_db()
                        name, confidence = new_name, 1.0 
                        self.logger.info(f"Added new face automatically: {new_name}")
                        
                        self.update_current_face(new_name, face_embedding, face_image)
                        
                        self.write_current_user_id(new_name, is_new_face=True)
                    
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
        """
        self.speaker_face_mapping[speaker_id] = person_name
        self._save_face_db()
        self.logger.info(f"Associated speaker {speaker_id} with person {person_name}")
    
    def get_person_from_speaker(self, speaker_id: str) -> str:
        """
        Get person name associated with a speaker ID.
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
        """
        try:
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                self.logger.error(f"Could not open video file: {video_file}")
                return diarized_segments
            
            if str(video_file).isdigit() or video_file in [0, 1]:
                if use_max_resolution:
                    resolutions_to_try = [
                        (4096, 2160),  
                        (3840, 2160), 
                        (2560, 1440),  
                        (1920, 1080), 
                        (1280, 720),  
                    ]
                    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    best_width, best_height = original_width, original_height
                    
                    for width, height in resolutions_to_try:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        
                        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        if actual_width * actual_height > best_width * best_height:
                            best_width, best_height = actual_width, actual_height
                    
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_height)
                    self.logger.info(f"Set camera to best available resolution: {best_width}x{best_height}")
                else:
                    width, height = camera_resolution
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    self.logger.info(f"Set camera resolution to {width}x{height}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.logger.info(f"Video properties: {actual_width}x{actual_height} at {fps} FPS, {total_frames} frames")
            
            face_detected = False
            detected_person = None
            max_attempts = 10 
            attempts = 0
            
            face_detections = [] 
            required_detections = 3  
            while len(face_detections) < required_detections and attempts < max_attempts:
                attempts += 1
                
                if attempts == 1:
                    frame_pos = total_frames // 2
                elif attempts == 2:
                    frame_pos = total_frames // 3
                elif attempts == 3:
                    frame_pos = (total_frames * 2) // 3
                else:
                    frame_pos = int(np.random.uniform(0, total_frames))
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if not ret:
                    self.logger.warning(f"Failed to read frame (attempt {attempts}/{max_attempts})")
                    continue
                
                face_results = self.process_video_frame(frame)
                
                if face_results:
                    face_info = face_results[0]
                    detected_person = face_info["name"]
                    face_detections.append((detected_person, face_info["embedding"], frame))
                    self.logger.info(f"Detected person: {detected_person} (detection {len(face_detections)}/{required_detections})")
            
            if face_detections:
                if len(face_detections) >= required_detections * 0.7:  
                    person_counts = {}
                    for person, _, _ in face_detections:
                        person_counts[person] = person_counts.get(person, 0) + 1
                    
                    most_common_person = max(person_counts.items(), key=lambda x: x[1])[0]
                    
                    face_detected = True
                    detected_person = self.standardize_person_name(most_common_person, output_dir)
                    self.logger.info(f"Consistently detected person: {detected_person}")
                    
                    best_frame = None
                    best_embedding = None
                    for person, embedding, frame in face_detections:
                        if person == detected_person:
                            best_frame = frame
                            best_embedding = embedding
                            break
            
            if not face_detected:
                self.logger.warning("No face detected in video after multiple attempts")
                
                current_time = int(time.time())
                unknown_id = f"Person_{current_time}"
                
                for segment in diarized_segments:
                    segment["person"] = unknown_id
                
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                conversation_data = {
                    "timestamp": timestamp,
                    "segments": diarized_segments
                }
                
                if knowledge_base:
                    conversation_data["knowledge_base"] = knowledge_base
                
                if workflow_state:
                    conversation_data["workflow_state"] = workflow_state
                
                self.save_conversation_for_person(
                    person_name=unknown_id,
                    conversation_data=conversation_data,
                    base_dir=output_dir
                )
            
            cap.release()
            return diarized_segments
            
        except Exception as e:
            self.logger.error(f"Error processing conversation with video: {e}")
            import traceback
            traceback.print_exc()
            return diarized_segments
    
    def capture_from_webcam(self, name: str, duration: int = 5) -> bool:
        """
        Capture a person's face from webcam and add to database.
        """
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.logger.error("Could not open webcam")
                return False
            
            self.logger.info(f"Capturing face for {name}. Please look at the camera...")
            
            time.sleep(1)
            
            start_time = time.time()
            frames = []
            
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frames.append(frame)
                
                cv2.imshow("Capturing face...", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            if not frames:
                self.logger.error("No frames captured")
                return False
            
            best_face_frame = None
            best_face_size = 0
            
            for frame in frames:
                faces = self.app.get(frame)
                if faces:
                    for face in faces:
                        bbox = face.bbox
                        face_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if face_size > best_face_size:
                            best_face_size = face_size
                            best_face_frame = frame
            
            if best_face_frame is None:
                self.logger.error("No face detected during capture")
                return False
            
            return self.add_face(name, best_face_frame)
            
        except Exception as e:
            self.logger.error(f"Error capturing face from webcam: {e}")
            return False
    
    def should_recheck_face(self) -> bool:
        """
        Determine if it's time to recheck the face based on the interval.
        """
        current_time = time.time()
        if current_time - self.last_face_check_time >= FACE_RECHECK_INTERVAL:
            return True
        return False

    def update_current_face(self, face_name: str, face_embedding: np.ndarray, face_image: np.ndarray = None) -> None:
        """
        Update the current face for this conversation session.
        """
        if not face_name.startswith("Person_") and not face_name.startswith("Unknown_"):
            if not hasattr(self, '_session_timestamp'):
                self._session_timestamp = int(time.time())
            face_name = f"Person_{self._session_timestamp}"
        
        self.current_face_name = face_name
        self.current_face_embedding = face_embedding
        self.last_face_check_time = time.time()
        
        if face_name not in self.known_faces:
            self.known_faces[face_name] = face_embedding
            self.logger.info(f"Added new face '{face_name}' to known faces database")
        else:
            existing_embedding = self.known_faces[face_name]
            updated_embedding = 0.7 * existing_embedding + 0.3 * face_embedding
            updated_embedding = updated_embedding / np.linalg.norm(updated_embedding)
            self.known_faces[face_name] = updated_embedding
            self.logger.debug(f"Updated face embedding for '{face_name}'")
        
        if face_image is not None:
            self._save_face_image(face_name, face_image)
        
        self._save_face_db()

    def _save_face_image(self, person_name: str, face_image: np.ndarray) -> None:
        """
        Save a face image for a person. This is now the primary method for storing face data.
        """
        try:
            base_dir = os.path.join(os.getcwd(), "conversations")
            person_dir = os.path.join(base_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            face_path = os.path.join(person_dir, "face.jpg")
            cv2.imwrite(face_path, face_image)
            
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
        """
        if not hasattr(self, 'current_face_embedding') or self.current_face_embedding is None:
            return False
        
        similarity = np.dot(self.current_face_embedding, face_embedding) / (
            np.linalg.norm(self.current_face_embedding) * np.linalg.norm(face_embedding))
        
        self.logger.info(f"Face similarity score: {similarity:.4f} (threshold: {threshold})")
        return similarity > threshold

    def migrate_existing_data(self, base_dir: str = None) -> bool:
        """
        Migrate existing conversation data to the new person### format.
        """
        try:
            base_dir = base_dir or os.path.join(os.getcwd(), "conversations")
            if not os.path.exists(base_dir):
                self.logger.warning(f"Conversations directory {base_dir} does not exist. Nothing to migrate.")
                return False
            
            dirs_to_migrate = []
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path) and (item.startswith("Person_") or item.startswith("Unknown_")):
                    dirs_to_migrate.append(item)
            
            if not dirs_to_migrate:
                self.logger.info("No directories need migration.")
                return True
            
            self.logger.info(f"Found {len(dirs_to_migrate)} directories to migrate")
            
            migration_map = {}
            next_person_number = 1
            
            for old_dir in dirs_to_migrate:
                new_dir = f"person{next_person_number:03d}"
                migration_map[old_dir] = new_dir
                next_person_number += 1
            
            for old_dir, new_dir in migration_map.items():
                old_path = os.path.join(base_dir, old_dir)
                new_path = os.path.join(base_dir, new_dir)
                
                os.makedirs(new_path, exist_ok=True)
                
                for item in os.listdir(old_path):
                    old_item_path = os.path.join(old_path, item)
                    new_item_path = os.path.join(new_path, item)
                    
                    if item.startswith("conversation_") and item.endswith(".txt"):
                        with open(old_item_path, 'r') as src, open(new_item_path, 'w') as dst:
                            content = src.read()
                            content = content.replace(old_dir.upper(), new_dir.upper())
                            dst.write(content)
                            
                        history_path = os.path.join(new_path, "conversation_history.txt")
                        history_exists = os.path.exists(history_path)
                        
                        with open(history_path, 'a' if history_exists else 'w') as f:
                            if not history_exists:
                                f.write(f"CONVERSATION HISTORY FOR {new_dir.upper()}\n")
                                f.write("="*80 + "\n\n")
                            
                            with open(old_item_path, 'r') as src:
                                lines = src.readlines()
                                if len(lines) > 3: 
                                    f.write(f"\n--- MIGRATED FROM {old_dir} ---\n\n")
                                    f.writelines(lines[3:]) 
                    else:
                        if os.path.isfile(old_item_path):
                            with open(old_item_path, 'rb') as src, open(new_item_path, 'wb') as dst:
                                dst.write(src.read())
                
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
        """
        if person_name.startswith("Person_") or person_name.startswith("Unknown_"):
            return person_name
        
        if person_name.startswith("person") and person_name[6:].isdigit():
            if not hasattr(self, '_session_timestamp'):
                self._session_timestamp = int(time.time())
            
            new_name = f"Person_{self._session_timestamp}"
            self.logger.info(f"Converting {person_name} to timestamp format: {new_name}")
            
            if person_name in self.known_faces:
                self.known_faces[new_name] = self.known_faces[person_name]
                del self.known_faces[person_name]
                self._save_face_db()
            
            return new_name
        
        if not hasattr(self, '_session_timestamp'):
            self._session_timestamp = int(time.time())
        
        new_name = f"Person_{self._session_timestamp}"
        self.logger.info(f"Created timestamp-based name: {new_name} for {person_name}")
        
        return new_name 

    def add_to_face_gallery(self, person_name: str, face_embedding: np.ndarray) -> None:
        """
        Add a face embedding to a person's gallery for better recognition.
        """
        if not hasattr(self, 'face_galleries'):
            self.face_galleries = {}
        
        if person_name not in self.face_galleries:
            self.face_galleries[person_name] = []
        
        self.face_galleries[person_name].append(face_embedding)
        if len(self.face_galleries[person_name]) > 5:
            self.face_galleries[person_name].pop(0) 
        
        self._save_face_galleries()

    def _load_face_galleries(self) -> None:
        """
        Load face galleries by scanning for multiple face images per person.
        """
        self.face_galleries = {}
        
        try:
            base_dir = os.path.join(os.getcwd(), "conversations")
            if not os.path.exists(base_dir):
                return
            
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                
                if not os.path.isdir(item_path):
                    continue
                
                face_images = []
                for file in os.listdir(item_path):
                    if (file.startswith("face_") and file.endswith(".jpg")) or file == "face.jpg":
                        face_path = os.path.join(item_path, file)
                        face_images.append(face_path)
                
                if face_images:
                    gallery = []
                    
                    for face_path in face_images:
                        try:
                            face_image = cv2.imread(face_path)
                            if face_image is None:
                                continue
                            
                            faces = self.app.get(face_image)
                            if not faces:
                                continue
                            
                            if len(faces) > 1:
                                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                            
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
        """
        best_match = "unknown"
        best_similarity = 0.0
        
        for name, known_embedding in self.known_faces.items():
            similarity = np.dot(face_embedding, known_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        if hasattr(self, 'face_galleries') and self.face_galleries:
            for name, gallery in self.face_galleries.items():
                if not gallery:
                    continue
                
                gallery_similarities = [np.dot(face_embedding, gallery_face) for gallery_face in gallery]
                
                gallery_best = max(gallery_similarities)
                
                if gallery_best > best_similarity:
                    best_similarity = gallery_best
                    best_match = name
        
        if best_similarity >= 0.25:
            return best_match, float(best_similarity)
        else:
            return "unknown", float(best_similarity)

    def get_persistent_identity(self, face_embedding: np.ndarray) -> str:
        """
        Get a persistent identity for a face embedding by comparing with all known faces.
        This helps maintain the same person ID across different sessions.
        """
        best_match = None
        best_similarity = 0.0
        
        for name, known_embedding in self.known_faces.items():
            similarity = np.dot(face_embedding, known_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        if hasattr(self, 'face_galleries') and self.face_galleries:
            for name, gallery in self.face_galleries.items():
                if not gallery:
                    continue
                
                gallery_similarities = [np.dot(face_embedding, gallery_face) for gallery_face in gallery]
                
                gallery_best = max(gallery_similarities)
                
                if gallery_best > best_similarity:
                    best_similarity = gallery_best
                    best_match = name
        
        if best_similarity >= 0.15 and best_match:
            self.logger.info(f"Found persistent identity: {best_match} with similarity {best_similarity:.2f}")
            return best_match
        
        if not hasattr(self, '_session_timestamp'):
            self._session_timestamp = int(time.time())
        
        new_id = f"Person_{self._session_timestamp}"
        self.logger.info(f"Created new persistent identity: {new_id}")
        return new_id

    def find_person_by_face(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Find a person by comparing a face image with all saved face images.
        """
        try:
            faces = self.app.get(face_image)
            
            if not faces:
                return "unknown", 0.0
            
            if len(faces) > 1:
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            
            face_embedding = faces[0].normed_embedding
            
            best_match = "unknown"
            best_similarity = 0.0
            
            for name, known_embedding in self.known_faces.items():
                similarity = np.dot(face_embedding, known_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
            
            if hasattr(self, 'face_galleries') and self.face_galleries:
                for name, gallery in self.face_galleries.items():
                    if not gallery:
                        continue
                    
                    gallery_similarities = [np.dot(face_embedding, gallery_face) for gallery_face in gallery]
                    
                    gallery_best = max(gallery_similarities)
                    
                    if gallery_best > best_similarity:
                        best_similarity = gallery_best
                        best_match = name
            
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
            system_dir = os.path.join(os.getcwd(), "conversations", "system_data")
            os.makedirs(system_dir, exist_ok=True)
            
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
        """
        try:
            debug_img = frame.copy()
            
            for i, face in enumerate(faces):
                bbox = face.bbox.astype(int)
                cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                confidence = f"Score: {face.det_score:.4f}"
                cv2.putText(debug_img, confidence, (bbox[0], bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if hasattr(face, 'kps') and face.kps is not None:
                    for kp in face.kps:
                        kx, ky = kp
                        cv2.circle(debug_img, (int(kx), int(ky)), 2, (255, 0, 0), 2)
            
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
        Main entry point for the new face recognition flow
        """
        faces_dir = os.path.join(os.getcwd(), "conversations", "faces")
        os.makedirs(faces_dir, exist_ok=True)
        
        faces = self.app.get_with_multiple_sizes(frame)
        
        if not faces or len(faces) == 0:
            return None, False
        
        self._save_debug_face_image(frame, faces, "input")
        
        largest_face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
        face_embedding = largest_face.embedding
        
        
        temp_dir = os.path.join(os.getcwd(), "temp_files")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"temp_face_{int(time.time())}.jpg")
        
        bbox = largest_face.bbox.astype(int)
        margin = 50 
        x1 = max(0, bbox[0] - margin)
        y1 = max(0, bbox[1] - margin)
        x2 = min(frame.shape[1], bbox[2] + margin)
        y2 = min(frame.shape[0], bbox[3] + margin)
        face_img = frame[y1:y2, x1:x2]
        cv2.imwrite(temp_file_path, face_img)
        
        
        existing_faces = [f for f in os.listdir(faces_dir) if f.startswith("face_") and f.endswith(".jpg")]
        
        if not existing_faces:
            face_id = f"face_{int(time.time())}"
            permanent_path = os.path.join(faces_dir, f"{face_id}.jpg")
            os.rename(temp_file_path, permanent_path)
            print(f"Nothing found adding temp file as: {face_id}")
            self.logger.info(f"Added first face with ID: {face_id}")
            
            self.write_current_user_id(face_id, is_new_face=True)
            
            return face_id, True
        
        best_match = None
        best_similarity = 0
        all_similarities = []
        
        for face_file in existing_faces:
            face_path = os.path.join(faces_dir, face_file)
            try:
                img = cv2.imread(face_path)
                if img is None:
                    print(f"Failed to load image: {face_path}")
                    continue
                    
                faces_in_img = self.app.get_with_multiple_sizes(img)
                
                if not faces_in_img or len(faces_in_img) == 0:
                    print(f"No face detected in {face_file}")
                    continue
                
                self._save_debug_face_image(img, faces_in_img, f"compare_{face_file.split('.')[0]}")
                
                existing_embedding = faces_in_img[0].embedding
                similarity = self._calculate_similarity(face_embedding, existing_embedding)
                all_similarities.append((face_file, similarity))
                
                
                if similarity > self.recognition_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = face_file.split(".")[0]  
                    print(f"New best match: {best_match} with similarity: {best_similarity:.4f}")
            except Exception as e:
                            print(f"Error processing face file {face_file}: {e}")
        
        if best_match:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            self.logger.info(f"Matched existing face: {best_match} (similarity: {best_similarity:.4f})")
            
            self.write_current_user_id(best_match, is_new_face=False)
            
            return best_match, False
        else:
            face_id = f"face_{int(time.time())}"
            permanent_path = os.path.join(faces_dir, f"{face_id}.jpg")
            os.rename(temp_file_path, permanent_path)
            self.logger.info(f"Added new face with ID: {face_id}")
            
            self.write_current_user_id(face_id, is_new_face=True)
            
            return face_id, True
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two face embeddings
        """
        try:
            if embedding1 is None or embedding2 is None:
                print("Warning: Received None embedding in similarity calculation")
                return 0
                
            if np.isnan(embedding1).any() or np.isnan(embedding2).any():
                print("Warning: NaN values in embeddings")
                return 0
                
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                print("Warning: Zero norm detected in embeddings")
                return 0
                
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            similarity = float(max(0, min(1, similarity)))
            
            return similarity
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0

    def write_current_user_id(self, face_id: str, is_new_face: bool = None) -> None:
        """
        Writes just the numeric part of the face ID to current_user_id.txt.
        Also indicates if this is a new face (needs upload) or existing face (skip upload).
        """
        try:
            if face_id is None:
                print("No face ID to write to current_user_id.txt")
                return
                
            if "_" in face_id:
                user_id = face_id.split("_")[1]  
            else:
                user_id = face_id
            
            output_file = os.path.join(os.getcwd(), "current_user_id.txt")
            
            if is_new_face is False:  
                output_text = f"skip_upload:{user_id}"
            else:
                output_text = user_id
                
            with open(output_file, 'w') as f:  # 'w' mode overwrites existing content
                f.write(output_text)
            
        except Exception as e:
            print(f"Error writing current user ID to file: {e}")

    def clear_current_user_id(self) -> None:
        """
        Clears the contents of the current_user_id.txt file or creates it if it doesn't exist.
        This is called at initialization to ensure a clean state for each new run.
        """
        try:
            output_file = os.path.join(os.getcwd(), "current_user_id.txt")
            with open(output_file, 'w') as f:
                f.write("NO_FACE_DETECTED")
        except Exception as e:
            print(f"Error clearing current_user_id.txt: {e}")

    @staticmethod
    def should_upload_to_supabase() -> Tuple[bool, str]:
        """
        Static helper method to determine if the current face should be uploaded to Supabase.
        """
        try:
            output_file = os.path.join(os.getcwd(), "current_user_id.txt")
            if not os.path.exists(output_file):
                return False, None
                
            with open(output_file, 'r') as f:
                content = f.read().strip()
                
            if content == "NO_FACE_DETECTED" or not content:
                return False, None
                
            if content.startswith("skip_upload:"):
                face_id = content.split("skip_upload:")[1]
                return False, face_id
                
            return True, content
            
        except Exception as e:
            print(f"Error checking if face should be uploaded: {e}")
            return False, None
