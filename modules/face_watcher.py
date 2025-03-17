import os
import time
import json
import traceback
import threading
import queue
import glob
from modules.supabase_integration import update_chat_history_in_supabase, upload_face_to_supabase

class FaceDirectoryWatcher:
    """
    Watches the conversations/faces directory for new images and uploads them to Supabase.
    """
    def __init__(self):
        self.watch_dir = os.path.join(os.getcwd(), "conversations", "faces")
        self.processed_files = set()
        self.running = False
        self.thread = None
        self.file_queue = queue.Queue()
        
        os.makedirs(self.watch_dir, exist_ok=True)
        
        self._scan_existing_files()
    
    def _scan_existing_files(self):
        """Scan for existing files and add them to the processed list"""
        for file_path in glob.glob(os.path.join(self.watch_dir, "face_*.jpg")):
            self.processed_files.add(os.path.basename(file_path))
            print(f"Added existing face file to processed list: {os.path.basename(file_path)}")
    
    def start(self):
        """Start the directory watcher thread"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._watch_directory)
        self.thread.daemon = True
        self.thread.start()
        
        worker_thread = threading.Thread(target=self._process_queue)
        worker_thread.daemon = True
        worker_thread.start()
        
        self._process_unprocessed_files()
    
    def _process_unprocessed_files(self):
        """Check for any unprocessed files in the user-history database and sync chat history"""
        from modules.supabase_integration import supabase_client
        
        if not supabase_client:
            print("Supabase client not initialized. Cannot check for unprocessed files.")
            return
        
        try:
            
            face_files = {}
            for file_path in glob.glob(os.path.join(self.watch_dir, "face_*.jpg")):
                filename = os.path.basename(file_path)
                user_id = filename[5:-4]  
                face_files[user_id] = file_path
            
            if not face_files:
                print("No face files found in the directory")
                return
            
            try:
                existing_users = (
                    supabase_client.table("user-history")
                    .select("user_id, profile_pic")
                    .execute()
                )
                
                existing_user_dict = {}
                for user in existing_users.data:
                    existing_user_dict[user["user_id"]] = user.get("profile_pic", "")
                
                print(f"Found {len(existing_user_dict)} existing users in the database")
                
                for user_id, file_path in face_files.items():
                    if user_id not in existing_user_dict:
                        self.file_queue.put(file_path)
                    elif not existing_user_dict[user_id]:
                        self.file_queue.put(file_path)
                    else:
                        expected_path = f"faces/face_{user_id}.jpg"
                        if expected_path not in existing_user_dict[user_id]:
                            self.file_queue.put(file_path)
                        else:
                            print(f"Skipping user ID {user_id}, already has correct profile pic")
                    
                    self._sync_chat_history_files(user_id)
                
            except Exception as e:
                print(f"Error checking existing users: {e}")
                traceback.print_exc()
                
                for user_id, file_path in face_files.items():
                    try:
                        user_check = (
                            supabase_client.table("user-history")
                            .select("user_id, profile_pic")
                            .eq("user_id", user_id)
                            .execute()
                        )
                        
                        if not user_check.data:
                            self.file_queue.put(file_path)
                        elif not user_check.data[0].get("profile_pic"):
                            self.file_queue.put(file_path)
                    except:
                        print(f"Skipping check for user ID {user_id} due to database errors")
        except Exception as e:
            print(f"Error processing unprocessed files: {e}")
            traceback.print_exc()
    
    def _sync_chat_history_files(self, user_id, specific_timestamp=None):
        """
        Sync chat history files for a specific user ID to the Supabase chat-history table.
        If specific_timestamp is provided, find and upload that specific run's file.
        Otherwise, find and upload the most recent chat history file.
        """
        conv_dir = os.path.join(os.getcwd(), "conversations", f"conversation_{user_id}")
        if not os.path.exists(conv_dir):
            print(f"No conversation directory found for user ID: {user_id}")
            return
        
        chat_history_files = []
        
        if specific_timestamp:
            target_file = os.path.join(conv_dir, f"chat_history_{specific_timestamp}.json")
            if os.path.exists(target_file):
                chat_history_files.append(target_file)
                print(f"Found specific chat history file for run: {specific_timestamp}")
            else:
                for filename in os.listdir(conv_dir):
                    if filename.startswith('chat_history_') and filename.endswith('.json'):
                        if specific_timestamp in filename:
                            chat_history_files.append(os.path.join(conv_dir, filename))
            
            if chat_history_files:
                print(f"Found {len(chat_history_files)} files containing timestamp: {specific_timestamp}")
        
        if not chat_history_files:
            for filename in os.listdir(conv_dir):
                if filename.startswith('chat_history_') and filename.endswith('.json'):
                    chat_history_files.append(os.path.join(conv_dir, filename))
        
        if not chat_history_files:
            print(f"No chat history files found for user ID: {user_id}")
            return
        
        if specific_timestamp and len(chat_history_files) == 1:
            target_file = chat_history_files[0]
            print(f"Uploading specified chat history file for run: {specific_timestamp}")
        else:
            chat_history_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
            target_file = chat_history_files[0]
            
            file_basename = os.path.basename(target_file)
            file_timestamp = "unknown"
            if file_basename.startswith("chat_history_") and file_basename.endswith(".json"):
                file_timestamp = file_basename[13:-5]
            
            print(f"Uploading chat history file: {file_basename} (timestamp: {file_timestamp})")
        
        def upload_thread():
            try:
                update_chat_history_in_supabase(user_id, target_file)
            except Exception as e:
                print(f"Error uploading chat history for user ID {user_id}: {e}")
                traceback.print_exc()
        
        thread = threading.Thread(target=upload_thread)
        thread.daemon = True
        thread.start()

    def stop(self):
        """Stop the directory watcher thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _watch_directory(self):
        """Watch the directory for new files"""
        while self.running:
            try:
                for file_path in glob.glob(os.path.join(self.watch_dir, "face_*.jpg")):
                    filename = os.path.basename(file_path)
                    if filename not in self.processed_files:
                        print(f"Found new face image: {filename}")
                        self.file_queue.put(file_path)
                        self.processed_files.add(filename)
            except Exception as e:
                print(f"Error in directory watcher: {e}")
            
            time.sleep(2.0)

    def _process_queue(self):
        """Process the queue of files to upload"""
        while self.running:
            try:
                try:
                    file_path = self.file_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                self._process_file(file_path)
                
                self.file_queue.task_done()
            except Exception as e:
                print(f"Error in queue processor: {e}")

    def _process_file(self, file_path):
        """Process a single file"""
        try:
            filename = os.path.basename(file_path)
            
            if filename.startswith("face_") and filename.endswith(".jpg"):
                user_id = filename[5:-4]  
                
                upload_face_to_supabase(file_path, user_id)
            else:
                print(f"Skipping file with invalid format: {filename}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            traceback.print_exc() 