#!/usr/bin/env python3

import os
import time
import numpy as np
import logging

class PersonFolderManager:
    """
    Centralized manager for person folders to ensure consistency across modules.
    """
    
    def __init__(self, base_dir=None):
        """
        Initialize the person folder manager.
        """
        self.base_dir = base_dir or os.path.join(os.getcwd(), "conversations")
        os.makedirs(self.base_dir, exist_ok=True)
        self.logger = logging.getLogger("PersonFolderManager")
        
    def get_person_folder(self, person_name, create_if_missing=True):
        """
        Get the folder path for a person, creating it if needed.
        """
        existing_folder = self.find_existing_person_folder(person_name)
        if existing_folder:
            person_name = existing_folder
        
        person_dir = os.path.join(self.base_dir, person_name)
        if create_if_missing:
            os.makedirs(person_dir, exist_ok=True)
            
        return person_dir
    
    def find_existing_person_folder(self, person_name):
        """
        Find an existing folder for a recognized person.
        """
        try:
            if not os.path.exists(self.base_dir):
                return None
            
            exact_path = os.path.join(self.base_dir, person_name)
            if os.path.exists(exact_path) and os.path.isdir(exact_path):
                self.logger.info(f"Found exact matching folder: {exact_path}")
                return person_name
            
            if "_" in person_name:
                base_name = person_name.split("_")[0]  
                
                matching_dirs = []
                for item in os.listdir(self.base_dir):
                    item_path = os.path.join(self.base_dir, item)
                    if os.path.isdir(item_path) and item.startswith(base_name):
                        # Check if this directory has conversation files
                        if (os.path.exists(os.path.join(item_path, "conversation_history.txt")) or 
                            os.path.exists(os.path.join(item_path, "conversation.txt")) or
                            os.path.exists(os.path.join(item_path, "conversation_1.txt"))):
                            matching_dirs.append(item)
                
                self.logger.info(f"Found {len(matching_dirs)} potential matching directories: {matching_dirs}")
                
                if matching_dirs:
                    face_embedding_path = os.path.join(self.base_dir, person_name, "face_embedding.npy")
                    if os.path.exists(face_embedding_path):
                        try:
                            current_embedding = np.load(face_embedding_path)
                            
                            best_match = None
                            best_similarity = 0.0
                            
                            for dir_name in matching_dirs:
                                dir_embedding_path = os.path.join(self.base_dir, dir_name, "face_embedding.npy")
                                if os.path.exists(dir_embedding_path):
                                    dir_embedding = np.load(dir_embedding_path)
                                    similarity = np.dot(current_embedding, dir_embedding)
                                    
                                    if similarity > best_similarity:
                                        best_similarity = similarity
                                        best_match = dir_name
                        
                            if best_match and best_similarity > 0.7: 
                                self.logger.info(f"Found face match in folder {best_match} with similarity {best_similarity:.2f}")
                                return best_match
                        except Exception as e:
                            self.logger.error(f"Error comparing face embeddings: {e}")
                    
                    self.logger.info(f"Using most recent directory: {matching_dirs[-1]}")
                    return matching_dirs[-1]
            
            return None
        except Exception as e:
            self.logger.error(f"Error finding existing person folder: {e}")
            return None
    
    def save_face_embedding(self, person_name, embedding):
        """
        Save face embedding to the person's folder.
        """
        try:
            person_dir = self.get_person_folder(person_name)
            
            embedding_path = os.path.join(person_dir, "face_embedding.npy")
            np.save(embedding_path, embedding)
            self.logger.info(f"Saved face embedding for {person_name}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving face embedding: {e}")
            return False
    
    def get_face_embedding(self, person_name):
        """
        Get face embedding from the person's folder.
        """
        try:
            person_dir = self.get_person_folder(person_name, create_if_missing=False)
            if not person_dir:
                return None
                
            embedding_path = os.path.join(person_dir, "face_embedding.npy")
            if os.path.exists(embedding_path):
                return np.load(embedding_path)
                
            return None
        except Exception as e:
            self.logger.error(f"Error getting face embedding: {e}")
            return None

    def save_knowledge_base(self, person_name, knowledge_base):
        """Save knowledge base to the person's folder."""
        person_dir = self.get_person_folder(person_name)
        kb_path = os.path.join(person_dir, "knowledge_base.txt")
        
        with open(kb_path, "w") as f:
            f.write(f"KNOWLEDGE BASE FOR {person_name.upper()}\n")
            f.write("="*80 + "\n\n")
            
            for topic, snippets in knowledge_base.items():
                f.write(f"TOPIC: {topic}\n")
                f.write("-"*80 + "\n")
                for snippet in snippets:
                    f.write(f"• {snippet}\n\n")
                f.write("\n")
        
        return kb_path

    def get_consistent_person_id(self, person_name):
        """
        Get a consistent person ID for a given person name.
        This ensures we use the same ID throughout the conversation.
        """
        existing_folder = self.find_existing_person_folder(person_name)
        if existing_folder:
            return existing_folder
        
        person_dir = self.get_person_folder(person_name)
        return os.path.basename(person_dir)

    def save_conversation_file(self, person_name, conversation_text, file_number=None):
        """
        Save a conversation to a numbered file in the person's folder.
        """
        try:
            person_dir = self.get_person_folder(person_name)
            
            if file_number is None:
                existing_files = []
                for item in os.listdir(person_dir):
                    if item.startswith("conversation_") and item.endswith(".txt"):
                        try:
                            num_part = item.replace("conversation_", "").replace(".txt", "")
                            if num_part.isdigit():
                                existing_files.append(int(num_part))
                        except:
                            pass
                
                if existing_files:
                    file_number = max(existing_files) + 1
                else:
                    file_number = 1
            
            file_path = os.path.join(person_dir, f"conversation_{file_number}.txt")
            
            self.logger.info(f"Saving conversation {file_number} for {person_name} to {file_path}")
            
            with open(file_path, "w") as f:
                f.write(f"CONVERSATION {file_number} WITH {person_name}\n")
                f.write("="*80 + "\n\n")
                f.write(f"DATE: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(conversation_text)
            
            self.logger.info(f"Successfully saved conversation {file_number} for {person_name}")
            return file_path
        except Exception as e:
            self.logger.error(f"Error saving conversation file: {e}")
            import traceback
            traceback.print_exc()
            return None
