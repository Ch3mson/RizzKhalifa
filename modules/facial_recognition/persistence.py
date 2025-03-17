"""
Persistence utilities for the facial recognition module.
Handles loading and saving of face data, identity mappings and face galleries.
"""

import os
import json
import numpy as np
import cv2
import time
from typing import Dict, List, Tuple, Optional, Set
import logging

class FacialRecognitionPersistence:
    """
    Handles persistence of facial recognition data including:
    - Face database loading/saving
    - Face galleries management
    - Identity mappings persistence
    - Face image storage
    """
    
    def __init__(self, face_db_path=None):
        """
        Initialize the persistence component.
        
        Args:
            face_db_path: Path to the face database file
        """
        self.logger = logging.getLogger("facial_recognition.persistence")
        
        project_dir = os.getcwd()
        
        if not face_db_path:
            system_dir = os.path.join(project_dir, "conversations", "system_data")
            os.makedirs(system_dir, exist_ok=True)
            face_db_path = os.path.join(system_dir, "speaker_mapping.json")
        
        self.face_db_path = face_db_path
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.face_db_path), exist_ok=True)
        
        # Initialize data structures
        self.known_faces = {}
        self.speaker_face_mapping = {}
        self.identity_mappings = {}
        self.face_galleries = {}
        self.persistent_identities = {}
        
    def load_face_db(self) -> Dict:
        """
        Load the face database from disk.
        
        Returns:
            Dict containing the loaded face database
        """
        try:
            if os.path.exists(self.face_db_path):
                with open(self.face_db_path, 'r') as f:
                    data = json.load(f)
                    
                    self.known_faces = data.get('known_faces', {})
                    self.speaker_face_mapping = data.get('speaker_mapping', {})
                    
                    # Convert face embeddings back to numpy arrays
                    for name, face_data in self.known_faces.items():
                        if isinstance(face_data, dict) and 'embedding' in face_data:
                            embedding = face_data['embedding']
                            if isinstance(embedding, list):
                                self.known_faces[name]['embedding'] = np.array(embedding)
                                
                    self.logger.info(f"Loaded {len(self.known_faces)} known faces from {self.face_db_path}")
                    return self.known_faces
            else:
                self.logger.info(f"No face database found at {self.face_db_path}, creating a new one")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading face database: {e}")
            return {}
            
    def save_face_db(self) -> bool:
        """
        Save the face database to disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            data = {
                'known_faces': {},
                'speaker_mapping': self.speaker_face_mapping
            }
            
            # Convert numpy arrays to lists for JSON serialization
            for name, face_data in self.known_faces.items():
                data['known_faces'][name] = face_data.copy()
                if isinstance(face_data, dict) and 'embedding' in face_data:
                    embedding = face_data['embedding']
                    if isinstance(embedding, np.ndarray):
                        data['known_faces'][name]['embedding'] = embedding.tolist()
            
            with open(self.face_db_path, 'w') as f:
                json.dump(data, f)
                
            self.logger.info(f"Saved {len(self.known_faces)} known faces to {self.face_db_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving face database: {e}")
            return False
    
    def save_face_image(self, person_name: str, face_image: np.ndarray, base_dir: str = None) -> str:
        """
        Save a face image to disk.
        
        Args:
            person_name: Name of the person
            face_image: Face image data
            base_dir: Base directory to save the image
            
        Returns:
            str: Path to the saved image file
        """
        try:
            if base_dir is None:
                base_dir = os.path.join(os.getcwd(), "conversations", "faces")
            
            os.makedirs(base_dir, exist_ok=True)
            
            # Create a safe filename from the person name
            safe_name = person_name.replace(" ", "_").lower()
            timestamp = int(time.time())
            
            file_path = os.path.join(base_dir, f"{safe_name}_{timestamp}.jpg")
            
            # Save the face image
            cv2.imwrite(file_path, face_image)
            
            self.logger.info(f"Saved face image for {person_name} to {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"Error saving face image: {e}")
            return ""
    
    def load_face_galleries(self, base_dir: str = None) -> Dict:
        """
        Load face galleries from the conversations directory.
        These are collections of face embeddings organized by person.
        
        Args:
            base_dir: Base directory to load galleries from
            
        Returns:
            Dict: Loaded face galleries
        """
        if base_dir is None:
            base_dir = os.path.join(os.getcwd(), "conversations")
        
        self.face_galleries = {}
        
        try:
            # Scan all conversation_{id} directories
            for entry in os.listdir(base_dir):
                if entry.startswith("conversation_"):
                    person_id = entry.split("_")[1]
                    person_dir = os.path.join(base_dir, entry)
                    
                    # Look for face_gallery.json
                    gallery_path = os.path.join(person_dir, "face_gallery.json")
                    if os.path.exists(gallery_path):
                        try:
                            with open(gallery_path, 'r') as f:
                                gallery_data = json.load(f)
                                
                                embeddings = gallery_data.get("embeddings", [])
                                
                                # Convert embeddings to numpy arrays
                                numpy_embeddings = []
                                for embedding in embeddings:
                                    if isinstance(embedding, list):
                                        numpy_embeddings.append(np.array(embedding))
                                
                                self.face_galleries[person_id] = {
                                    "embeddings": numpy_embeddings,
                                    "count": len(numpy_embeddings)
                                }
                                
                                self.logger.info(f"Loaded {len(numpy_embeddings)} embeddings for person {person_id}")
                        except Exception as e:
                            self.logger.error(f"Error loading face gallery for {person_id}: {e}")
            
            return self.face_galleries
            
        except Exception as e:
            self.logger.error(f"Error loading face galleries: {e}")
            return {}
    
    def save_face_gallery(self, person_id: str, embeddings: List[np.ndarray], base_dir: str = None) -> bool:
        """
        Save a face gallery for a person.
        
        Args:
            person_id: ID of the person
            embeddings: List of face embeddings
            base_dir: Base directory to save the gallery
            
        Returns:
            bool: True if successful, False otherwise
        """
        if base_dir is None:
            base_dir = os.path.join(os.getcwd(), "conversations")
        
        try:
            # Ensure the conversation directory exists
            person_dir = os.path.join(base_dir, f"conversation_{person_id}")
            os.makedirs(person_dir, exist_ok=True)
            
            # Convert embeddings to lists for JSON serialization
            list_embeddings = []
            for embedding in embeddings:
                if isinstance(embedding, np.ndarray):
                    list_embeddings.append(embedding.tolist())
            
            # Save the gallery data
            gallery_path = os.path.join(person_dir, "face_gallery.json")
            with open(gallery_path, 'w') as f:
                json.dump({
                    "person_id": person_id,
                    "embeddings": list_embeddings,
                    "count": len(list_embeddings),
                    "last_updated": time.strftime('%Y-%m-%d %H:%M:%S')
                }, f)
            
            self.logger.info(f"Saved face gallery with {len(list_embeddings)} embeddings for person {person_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving face gallery for {person_id}: {e}")
            return False
    
    def load_identity_mappings(self, system_dir: str = None) -> Dict:
        """
        Load identity mappings from disk.
        
        Args:
            system_dir: Directory containing system data
            
        Returns:
            Dict: Loaded identity mappings
        """
        if system_dir is None:
            system_dir = os.path.join(os.getcwd(), "conversations", "system_data")
        
        try:
            mapping_path = os.path.join(system_dir, "identity_mappings.json")
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    self.identity_mappings = json.load(f)
                    self.logger.info(f"Loaded {len(self.identity_mappings)} identity mappings")
                    return self.identity_mappings
            else:
                self.logger.info("No identity mappings found")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading identity mappings: {e}")
            return {}
    
    def save_identity_mappings(self, system_dir: str = None) -> bool:
        """
        Save identity mappings to disk.
        
        Args:
            system_dir: Directory to save system data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if system_dir is None:
            system_dir = os.path.join(os.getcwd(), "conversations", "system_data")
        
        try:
            os.makedirs(system_dir, exist_ok=True)
            mapping_path = os.path.join(system_dir, "identity_mappings.json")
            
            with open(mapping_path, 'w') as f:
                json.dump(self.identity_mappings, f)
                
            self.logger.info(f"Saved {len(self.identity_mappings)} identity mappings")
            return True
        except Exception as e:
            self.logger.error(f"Error saving identity mappings: {e}")
            return False
    
    def load_persistent_identities(self, system_dir: str = None) -> Dict:
        """
        Load persistent identity data.
        
        Args:
            system_dir: Directory containing system data
            
        Returns:
            Dict: Loaded persistent identities
        """
        if system_dir is None:
            system_dir = os.path.join(os.getcwd(), "conversations", "system_data")
        
        try:
            identity_path = os.path.join(system_dir, "persistent_identities.json")
            if os.path.exists(identity_path):
                with open(identity_path, 'r') as f:
                    data = json.load(f)
                    
                    # Convert embeddings back to numpy arrays
                    for person_id, person_data in data.items():
                        if "reference_embedding" in person_data:
                            if isinstance(person_data["reference_embedding"], list):
                                data[person_id]["reference_embedding"] = np.array(
                                    person_data["reference_embedding"]
                                )
                    
                    self.persistent_identities = data
                    self.logger.info(f"Loaded {len(self.persistent_identities)} persistent identities")
                    return self.persistent_identities
            else:
                self.logger.info("No persistent identities found")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading persistent identities: {e}")
            return {}
    
    def save_persistent_identities(self, system_dir: str = None) -> bool:
        """
        Save persistent identity data.
        
        Args:
            system_dir: Directory to save system data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if system_dir is None:
            system_dir = os.path.join(os.getcwd(), "conversations", "system_data")
        
        try:
            os.makedirs(system_dir, exist_ok=True)
            identity_path = os.path.join(system_dir, "persistent_identities.json")
            
            # Prepare data for serialization
            data = {}
            for person_id, person_data in self.persistent_identities.items():
                data[person_id] = person_data.copy()
                if "reference_embedding" in person_data:
                    if isinstance(person_data["reference_embedding"], np.ndarray):
                        data[person_id]["reference_embedding"] = person_data["reference_embedding"].tolist()
            
            with open(identity_path, 'w') as f:
                json.dump(data, f)
                
            self.logger.info(f"Saved {len(self.persistent_identities)} persistent identities")
            return True
        except Exception as e:
            self.logger.error(f"Error saving persistent identities: {e}")
            return False
            
    def write_current_user_id(self, face_id: str) -> None:
        """
        Write the current user ID to a file so other processes can access it.
        
        Args:
            face_id: Face ID to write
        """
        try:
            user_id_path = os.path.join(os.getcwd(), "current_user_id.txt")
            with open(user_id_path, 'w') as f:
                f.write(face_id)
            self.logger.info(f"Wrote current user ID: {face_id}")
        except Exception as e:
            self.logger.error(f"Error writing current user ID: {e}")
    
    def clear_current_user_id(self) -> None:
        """
        Clear the current user ID file.
        """
        try:
            user_id_path = os.path.join(os.getcwd(), "current_user_id.txt")
            if os.path.exists(user_id_path):
                with open(user_id_path, 'w') as f:
                    f.write("")
                self.logger.info("Cleared current user ID")
        except Exception as e:
            self.logger.error(f"Error clearing current user ID: {e}") 