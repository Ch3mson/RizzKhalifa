"""
Integration module for facial recognition.
Provides functionality that integrates facial recognition with other systems like video processing.
"""

import os
import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import logging

class FaceVideoIntegration:
    """
    Integrates facial recognition with video processing and speaker diarization.
    """
    
    def __init__(self, face_recognizer, persistence_manager):
        """
        Initialize the integration module.
        
        Args:
            face_recognizer: Face recognition component
            persistence_manager: Persistence management component
        """
        self.logger = logging.getLogger("facial_recognition.integration")
        self.face_recognizer = face_recognizer
        self.persistence = persistence_manager
        self.speaker_face_mapping = {}
        
    def associate_speaker_with_face(self, speaker_id: str, person_name: str) -> None:
        """
        Associate a speaker ID with a face/person.
        
        Args:
            speaker_id: Speaker ID from diarization
            person_name: Person name/ID
        """
        self.speaker_face_mapping[speaker_id] = person_name
        self.logger.info(f"Associated speaker {speaker_id} with person {person_name}")
    
    def get_person_from_speaker(self, speaker_id: str) -> str:
        """
        Get person name/ID associated with a speaker.
        
        Args:
            speaker_id: Speaker ID from diarization
            
        Returns:
            str: Person name/ID or None if not found
        """
        return self.speaker_face_mapping.get(speaker_id)
    
    def process_video_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a video frame for face detection and recognition.
        
        Args:
            frame: Video frame to process
            
        Returns:
            Dict: Processing results with detected faces and identities
        """
        if frame is None:
            self.logger.error("Cannot process None frame")
            return {"error": "Invalid frame", "faces": []}
        
        # Detect faces in the frame
        faces = self.face_recognizer.detect_faces(frame)
        result = {"faces": []}
        
        # Process each detected face
        for i, face in enumerate(faces):
            face_bbox = face.bbox.astype(int)
            face_embedding = face.embedding
            
            # Try to recognize the face
            person_id = None
            similarity = 0.0
            
            # Check if it's the same as current tracked face
            if self.face_recognizer.current_face_embedding is not None:
                if self.face_recognizer.is_same_face(face_embedding):
                    person_id = self.face_recognizer.current_face_name
                    similarity = self.face_recognizer._calculate_similarity(
                        self.face_recognizer.current_face_embedding,
                        face_embedding
                    )
            
            # If not recognized as current face, try to match with known faces
            if not person_id:
                person_id, similarity = self.face_recognizer.recognize_face(
                    self.persistence.known_faces, 
                    face_embedding
                )
                
                # If recognized, update current face
                if person_id:
                    self.face_recognizer.update_current_face(
                        face_name=person_id,
                        face_embedding=face_embedding
                    )
                    
                    # Write current user ID to file for other processes
                    self.persistence.write_current_user_id(person_id)
            
            # Append face data to result
            face_data = {
                "index": i,
                "bbox": face_bbox.tolist(),
                "landmarks": face.landmark.tolist() if hasattr(face, "landmark") else None,
                "person_id": person_id,
                "similarity": float(similarity),
                "confidence": float(face.det_score) if hasattr(face, "det_score") else 0.0
            }
            
            result["faces"].append(face_data)
        
        # Update result with tracking info
        result["current_face"] = self.face_recognizer.current_face_name
        result["face_count"] = len(faces)
        
        return result
    
    def process_conversation_with_video(self, 
                                video_file: str, 
                                diarized_segments: List[Dict],
                                output_dir: str,
                                knowledge_base: Dict = None,
                                use_max_resolution: bool = True) -> List[Dict]:
        """
        Process a conversation with video, associating speakers with faces.
        
        Args:
            video_file: Path to the video file
            diarized_segments: List of diarized speech segments
            output_dir: Directory to save output files
            knowledge_base: Optional knowledge base to update
            use_max_resolution: Whether to use maximum resolution
            
        Returns:
            List[Dict]: Enhanced segments with face information
        """
        if not os.path.exists(video_file):
            self.logger.error(f"Video file not found: {video_file}")
            return diarized_segments
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video file
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            self.logger.error(f"Could not open video file: {video_file}")
            return diarized_segments
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Initialize speaker-face associations
        speaker_face_mappings = {}
        face_timestamps = {}
        
        # Process key frames to associate speakers with faces
        for segment in diarized_segments:
            # Skip segments without speaker
            if "speaker" not in segment:
                continue
                
            speaker = segment.get("speaker")
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", 0)
            
            # Skip if already mapped
            if speaker in speaker_face_mappings:
                continue
                
            # Sample frame at the middle of the segment
            mid_time = (start_time + end_time) / 2
            frame_idx = int(mid_time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # Process the frame to detect faces
            result = self.process_video_frame(frame)
            
            # If faces found, associate with speaker
            if result.get("faces"):
                # Use the face with highest confidence
                faces = sorted(result["faces"], key=lambda x: x.get("confidence", 0), reverse=True)
                
                for face in faces:
                    person_id = face.get("person_id")
                    
                    if person_id:
                        speaker_face_mappings[speaker] = person_id
                        face_timestamps[speaker] = mid_time
                        self.associate_speaker_with_face(speaker, person_id)
                        
                        self.logger.info(f"Associated speaker {speaker} with person {person_id}")
                        
                        # Save face image if not already saved
                        face_bbox = face.get("bbox")
                        if face_bbox:
                            x1, y1, x2, y2 = face_bbox
                            face_img = frame[y1:y2, x1:x2]
                            
                            if face_img.size > 0:
                                face_path = os.path.join(output_dir, f"face_{speaker}_{person_id}.jpg")
                                cv2.imwrite(face_path, face_img)
                        
                        break
        
        # Update segments with face information
        enhanced_segments = []
        for segment in diarized_segments:
            segment_copy = segment.copy()
            
            if "speaker" in segment:
                speaker = segment.get("speaker")
                if speaker in speaker_face_mappings:
                    segment_copy["person_id"] = speaker_face_mappings[speaker]
                    
                    # Add additional face info if available in knowledge base
                    if knowledge_base and speaker_face_mappings[speaker] in knowledge_base:
                        segment_copy["person_info"] = knowledge_base[speaker_face_mappings[speaker]]
            
            enhanced_segments.append(segment_copy)
        
        # Close video file
        cap.release()
        
        return enhanced_segments
    
    def save_debug_face_image(self, frame: np.ndarray, faces: list, prefix: str = "debug") -> str:
        """
        Save a debug image with face bounding boxes.
        
        Args:
            frame: Input frame
            faces: List of detected faces
            prefix: Filename prefix
            
        Returns:
            str: Path to the saved debug image
        """
        try:
            if frame is None or len(frame.shape) != 3:
                return ""
                
            debug_frame = frame.copy()
            
            # Draw face bounding boxes
            for i, face in enumerate(faces):
                if hasattr(face, 'bbox'):
                    bbox = face.bbox.astype(int)
                    confidence = face.det_score if hasattr(face, 'det_score') else 0.0
                    
                    # Calculate colors based on confidence (green for high confidence)
                    color = (0, int(255 * confidence), int(255 * (1 - confidence)))
                    
                    # Draw rectangle around face
                    cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    
                    # Draw face index
                    cv2.putText(
                        debug_frame, 
                        f"Face {i+1}", 
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                    )
                    
                    # If face has landmarks, draw them
                    if hasattr(face, 'landmark'):
                        landmark = face.landmark
                        for j in range(len(landmark)):
                            x, y = int(landmark[j][0]), int(landmark[j][1])
                            cv2.circle(debug_frame, (x, y), 2, (0, 255, 255), -1)
            
            # Add timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.jpg"
            
            # Ensure debug directory exists
            debug_dir = os.path.join(os.getcwd(), "temp_files", "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save the image
            output_path = os.path.join(debug_dir, filename)
            cv2.imwrite(output_path, debug_frame)
            
            return output_path
        except Exception as e:
            self.logger.error(f"Error saving debug image: {e}")
            return "" 