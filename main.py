#!/usr/bin/env python3

import sys
import argparse
import os
import traceback
import time
import json
from modules.assistant import ConversationAssistant
import threading
from supabase import create_client
from dotenv import load_dotenv
import queue
import glob

load_dotenv()

CURRENT_FACE_ID = None
last_state = None
CURRENT_RUN_TIMESTAMP = None

supabase_url = os.environ.get("SUPABASE_URL", "")
supabase_key = os.environ.get("SUPABASE_KEY", "")
supabase_client = None

if supabase_url and supabase_key:
    try:
        supabase_client = create_client(supabase_url, supabase_key)
        print("Supabase client initialized for face image uploads")
    except Exception as e:
        print(f"Error initializing Supabase client: {e}")
        supabase_client = None

def clean_supabase_duplicates():
    """
    Clean up any duplicate entries in the user-history table.
    Each user_id should have only one entry.
    """
    if not supabase_client:
        print("Supabase client not initialized. Cannot clean up duplicates.")
        return
    
    try:
        all_entries = (
            supabase_client.table("user-history")
            .select("id, user_id, created_at")
            .order("created_at")
            .execute()
        )
        
        if not all_entries.data:
            print("No entries found in user-history table.")
            return
        
        user_entries = {}
        for entry in all_entries.data:
            user_id = entry.get("user_id")
            if user_id not in user_entries:
                user_entries[user_id] = []
            user_entries[user_id].append(entry)
        
        duplicates_found = 0
        for user_id, entries in user_entries.items():
            if len(entries) > 1:
                duplicates_found += len(entries) - 1
                entries.sort(key=lambda x: x.get("created_at", ""))
                all_personal_info = []
                profile_pic = None
                
                for i, entry in enumerate(entries):
                    if i > 0:  
                        full_entry = (
                            supabase_client.table("user-history")
                            .select("*")
                            .eq("id", entry["id"])
                            .execute()
                        )
                        
                        if full_entry.data:
                            entry_data = full_entry.data[0]
                            if entry_data.get("personal_info"):
                                all_personal_info.extend(entry_data["personal_info"])
                            if entry_data.get("profile_pic") and not profile_pic:
                                profile_pic = entry_data["profile_pic"]
                        
                        delete_response = (
                            supabase_client.table("user-history")
                            .delete()
                            .eq("id", entry["id"])
                            .execute()
                        )
                        print(f"Deleted duplicate entry with ID: {entry['id']}")
                
                if all_personal_info or profile_pic:
                    kept_entry = entries[0]
                    
                    full_kept_entry = (
                        supabase_client.table("user-history")
                        .select("*")
                        .eq("id", kept_entry["id"])
                        .execute()
                    )
                    
                    if full_kept_entry.data:
                        kept_data = full_kept_entry.data[0]
                        update_data = {}
                        
                        if all_personal_info:
                            existing_personal_info = kept_data.get("personal_info", [])
                            for item in all_personal_info:
                                if item not in existing_personal_info:
                                    existing_personal_info.append(item)
                            update_data["personal_info"] = existing_personal_info
                        
                        if profile_pic and not kept_data.get("profile_pic"):
                            update_data["profile_pic"] = profile_pic
                        
                        if update_data:
                            update_data["updated_at"] = time.strftime('%Y-%m-%d %H:%M:%S')
                            update_response = (
                                supabase_client.table("user-history")
                                .update(update_data)
                                .eq("id", kept_entry["id"])
                                .execute()
                            )
            
    except Exception as e:
        print(f"Error cleaning up duplicates: {e}")
        traceback.print_exc()

def upload_face_to_supabase(face_path, user_id):
    """
    Upload a face image to Supabase and create an entry in the user-history bucket
    Ensures that only one entry per user_id exists in the user-history table
    """
    if not supabase_client:
        print("Supabase client not initialized. Cannot upload face.")
        return False
    
    try:
        file_name = os.path.basename(face_path)
        avatar_path = f"faces/{file_name}"
        
        with open(face_path, "rb") as f:
            file_content = f.read()
            
        bucket_response = supabase_client.storage.from_("avatars").upload(
            path=avatar_path,
            file=file_content,
            file_options={"content-type": "image/jpeg"},
        )
        
        print(f"Uploaded face image to Supabase: {avatar_path}")
        
        existing_data = (
            supabase_client.table("user-history")
            .select("id")
            .eq("user_id", user_id)
            .execute()
        )
        
        if existing_data.data and len(existing_data.data) > 1:
            for entry in existing_data.data:
                delete_response = (
                    supabase_client.table("user-history")
                    .delete()
                    .eq("id", entry["id"])
                    .execute()
                )
            existing_data.data = []
            
        if not existing_data.data or len(existing_data.data) == 0:
            db_response = (
                supabase_client.table("user-history")
                .insert({
                    "user_id": user_id,
                    "personal_info": [],
                    "profile_pic": avatar_path
                })
                .execute()
            )
        else:
            db_response = (
                supabase_client.table("user-history")
                .update({
                    "profile_pic": avatar_path
                })
                .eq("id", existing_data.data[0]["id"])
                .execute()
            )
        return True
    except Exception as e:
        print(f"Error uploading face to Supabase: {e}")
        traceback.print_exc()
        return False

def update_personal_info_in_supabase(user_id, personal_info):
    """
    Update the personal information for a user in the Supabase database.
    """
    if not supabase_client:
        print("Supabase client not initialized. Cannot update personal info.")
        return False
    
    try:
        existing_data = (
            supabase_client.table("user-history")
            .select("id")
            .eq("user_id", user_id)
            .execute()
        )
        
        if existing_data.data and len(existing_data.data) > 1:
            for i, entry in enumerate(existing_data.data):
                if i > 0: 
                    delete_response = (
                        supabase_client.table("user-history")
                        .delete()
                        .eq("id", entry["id"])
                        .execute()
                    )
            existing_data = (
                supabase_client.table("user-history")
                .select("id")
                .eq("user_id", user_id)
                .execute()
            )
        
        if existing_data.data and len(existing_data.data) == 1:
            db_response = (
                supabase_client.table("user-history")
                .update({
                    "personal_info": personal_info
                })
                .eq("id", existing_data.data[0]["id"])  
                .execute()
            )
            return True
        elif not existing_data.data or len(existing_data.data) == 0:
            db_response = (
                supabase_client.table("user-history")
                .insert({
                    "user_id": user_id,
                    "personal_info": personal_info
                })
                .execute()
            )
            return True
        else:
            return False
    except Exception as e:
        print(f"Error updating personal info in Supabase: {e}")
        traceback.print_exc()
        return False

def update_chat_history_in_supabase(user_id, chat_history_file):
    """
    Update the chat history for a user in the Supabase database.
    For each run, we create ONE entry in the chat-history table and update that same entry
    """
    if not supabase_client:
        print("Supabase client not initialized. Cannot update chat history.")
        return False
    
    try:
        filename = os.path.basename(chat_history_file)
        run_timestamp = ""
        if filename.startswith("chat_history_") and filename.endswith(".json"):
            run_timestamp = filename[13:-5]  
        else:
            run_timestamp = time.strftime('%Y%m%d_%H%M%S') 
        
        current_date = time.strftime('%Y-%m-%d')
        
        with open(chat_history_file, 'r') as f:
            chat_data = json.load(f)
        
        existing_entries = (
            supabase_client.table("chat-history")
            .select("id, created_at, chat_history")
            .eq("user_id", user_id)
            .execute()
        )
        
        wrapped_chat_data = {
            "metadata": {
                "run_id": run_timestamp,
                "updated_at": time.strftime('%Y-%m-%d %H:%M:%S')
            },
            "messages": chat_data  
        }
        
        matching_entry = None
        if existing_entries.data:
            for entry in existing_entries.data:
                entry_chat = entry.get("chat_history", {})
                
                if isinstance(entry_chat, dict) and entry_chat.get("metadata"):
                    if entry_chat.get("metadata", {}).get("run_id") == run_timestamp:
                        matching_entry = entry
                        break
                
                elif entry.get("created_at", "").startswith(current_date):
                    if (isinstance(entry_chat, list) and chat_data and 
                        len(entry_chat) > 0 and len(chat_data) > 0):
                        entry_first_msg = entry_chat[0].get("message", "") if isinstance(entry_chat[0], dict) else ""
                        our_first_msg = chat_data[0].get("message", "") if isinstance(chat_data[0], dict) else ""
                        
                        if entry_first_msg and our_first_msg and entry_first_msg == our_first_msg:
                            matching_entry = entry
                            break
        
        if not matching_entry and existing_entries.data:
            today_entries = [
                entry for entry in existing_entries.data 
                if entry.get("created_at", "").startswith(current_date)
            ]
            
            if today_entries:
                today_entries.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                matching_entry = today_entries[0]
        
        if matching_entry:
            entry_id = matching_entry["id"]
            db_response = (
                supabase_client.table("chat-history")
                .update({
                    "chat_history": wrapped_chat_data
                })
                .eq("id", entry_id)
                .execute()
            )
        else:
            db_response = (
                supabase_client.table("chat-history")
                .insert({
                    "user_id": user_id,
                    "chat_history": wrapped_chat_data
                })
                .execute()
            )
        return True
    
    except Exception as e:
        traceback.print_exc()
        return False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Voice Assistant with Speaker Diarization")
    
    parser.add_argument("--diarization", action="store_true", default=True,
                      help="Enable speaker diarization (default: enabled)")
    parser.add_argument("--no-diarization", action="store_false", dest="diarization",
                      help="Disable speaker diarization")
    parser.add_argument("--speakers", type=int, default=2,
                      help="Expected number of speakers in conversations (default: 2)")
    parser.add_argument("--screen", action="store_true", default=False,
                      help="Enable screen capture for automatic facial recognition (default: disabled)")
    parser.add_argument("--debug", action="store_true", default=False,
                      help="Enable debug mode with verbose logging")
    parser.add_argument("--test", action="store_true", default=False,
                      help="Run in test mode to validate workflow functionality")
    parser.add_argument("--face-recheck", type=int, default=60,
                      help="Interval in seconds to check if the detected face is the same (default: 60)")
    parser.add_argument("--keep-temp", action="store_true", default=False,
                      help="Keep temporary files after execution (default: False)")
    
    return parser.parse_args()

def load_existing_conversation_data(face_id):
    """
    Load existing conversation data for a recognized face.
    
    Args:
        face_id: The numeric ID of the detected face
        
    Returns:
        dict: The loaded conversation data or None if not found
    """
    try:
        json_path = os.path.join(os.getcwd(), "conversations", f"conversation_{face_id}", "conversation_data.json")
        
        if not os.path.exists(json_path):
            print(f"No existing conversation data found for face ID: {face_id}")
            return None
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
            chat_history_files = data.get("chat_history_files", [])
            
            if chat_history_files:
                
                chat_history_files.sort(reverse=True)
                most_recent = chat_history_files[0]
                most_recent_path = os.path.join(os.getcwd(), "conversations", f"conversation_{face_id}", most_recent)
                if os.path.exists(most_recent_path):
                    with open(most_recent_path, 'r') as f2:
                        try:
                            most_recent_data = json.load(f2)
                        except json.JSONDecodeError:
                            print(f"Error reading chat history file: {most_recent}")
            
            knowledge_base = data.get("knowledge_base", {})
            personal_info_json_path = os.path.join(os.getcwd(), "conversations", f"conversation_{face_id}", "personal_info.json")
            if os.path.exists(personal_info_json_path):
                try:
                    with open(personal_info_json_path, 'r') as f:
                        personal_info_data = json.load(f)
                        data["personal_info"] = personal_info_data
                        print(f"Loaded personal info with {len(personal_info_data)} items")
                except json.JSONDecodeError:
                    print("Error reading personal info JSON file")
            
            kb = data.get("knowledge_base", {})
            personal_info = data.get("personal_info", [])
            
            return data
    except Exception as e:
        print(f"Error loading conversation data: {e}")
        return None

def detect_and_recognize_face(assistant):
    """
    Captures a single frame, detects a face, and compares it with stored faces.
    """
    global CURRENT_FACE_ID, last_state
    
    if not assistant.use_camera or not assistant.facial_recognition:
        print("Camera or facial recognition is not enabled. Skipping face detection.")
        return None
    frame = assistant._capture_screen_frame()
    if frame is None:
        print("Error: Failed to capture frame for face detection.")
        return None
    
    try:
        print("Calling InsightFace facial recognition...")
        face_id, is_new_face = assistant.facial_recognition.manage_face_recognition(frame)
        
        if face_id:
            numeric_id = face_id.split('_')[1]
            CURRENT_FACE_ID = numeric_id
            print(f"Setting current face ID to: {CURRENT_FACE_ID}")
            
            init_conversation_directory(numeric_id)
            
            if is_new_face:
                face_path = os.path.join(os.getcwd(), "conversations", "faces", f"{face_id}.jpg")
                if os.path.exists(face_path):
                    upload_thread = threading.Thread(
                        target=upload_face_to_supabase,
                        args=(face_path, numeric_id)
                    )
                    upload_thread.daemon = True
                    upload_thread.start()
            
            if not is_new_face:
                existing_data = load_existing_conversation_data(numeric_id)
                
                if existing_data and hasattr(assistant, 'workflow'):
                    try:
                        if not hasattr(assistant.workflow, 'state') or not assistant.workflow.state:
                            assistant.workflow.state = {}
                        
                        if "chat_history" in existing_data and existing_data["chat_history"]:
                            assistant.workflow.state["chat_history"] = existing_data["chat_history"]
                        
                        if "knowledge_base" in existing_data and existing_data["knowledge_base"]:
                            assistant.workflow.state["knowledge_base"] = existing_data["knowledge_base"]
                        
                        if "personal_info" in existing_data and existing_data["personal_info"]:
                            assistant.workflow.state["personal_info"] = existing_data["personal_info"]
                        
                        last_state = assistant.workflow.state.copy() if assistant.workflow.state else None
                        
                    except Exception as e:
                        print(f"Error restoring conversation state: {e}")
                        traceback.print_exc()
            
            assistant.facial_recognition.update_current_face(
                face_name=face_id,
                face_embedding=None  
            )
            
            return face_id
        else:
            for _ in range(3): 
                time.sleep(1)  
                frame = assistant._capture_screen_frame()
                if frame is not None:
                    try:
                        face_id, is_new_face = assistant.facial_recognition.manage_face_recognition(frame)
                        if face_id:
                            numeric_id = face_id.split('_')[1]
                            CURRENT_FACE_ID = numeric_id
                            
                            init_conversation_directory(numeric_id)
                            
                            if not is_new_face:
                                existing_data = load_existing_conversation_data(numeric_id)
                                
                                if existing_data and hasattr(assistant, 'workflow'):
                                    try:
                                        if not hasattr(assistant.workflow, 'state') or not assistant.workflow.state:
                                            assistant.workflow.state = {}
                                        
                                        if "chat_history" in existing_data and existing_data["chat_history"]:
                                            assistant.workflow.state["chat_history"] = existing_data["chat_history"]
                                        
                                        if "knowledge_base" in existing_data and existing_data["knowledge_base"]:
                                            assistant.workflow.state["knowledge_base"] = existing_data["knowledge_base"]
                                        
                                        if "personal_info" in existing_data and existing_data["personal_info"]:
                                            assistant.workflow.state["personal_info"] = existing_data["personal_info"]
                                        
                                        last_state = assistant.workflow.state.copy() if assistant.workflow.state else None
                                        
                                    except Exception as e:
                                        traceback.print_exc()
                            
                            assistant.facial_recognition.update_current_face(
                                face_name=face_id,
                                face_embedding=None
                            )
                            return face_id
                    except Exception as e:
                        print(f"Error restoring conversation state: {e}")
                        traceback.print_exc()
            return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def init_conversation_directory(face_id):
    """
    Initialize the conversation directory structure for a specific face ID.
    Creates a conversations/conversation_{id} directory with necessary files.
    """
    global CURRENT_RUN_TIMESTAMP
    
    try:
        CURRENT_RUN_TIMESTAMP = time.strftime('%Y%m%d_%H%M%S')
        
        conv_dir = os.path.join(os.getcwd(), "conversations", f"conversation_{face_id}")
        os.makedirs(conv_dir, exist_ok=True)
        
        chat_history_path = os.path.join(conv_dir, f"chat_history_{CURRENT_RUN_TIMESTAMP}.json")
        
        with open(chat_history_path, 'w') as f:
            json.dump([], f, indent=2)
            
        files = [
            "conversation_data.json"
        ]
        
        personal_info_json_path = os.path.join(conv_dir, "personal_info.json")
        if not os.path.exists(personal_info_json_path):
            with open(personal_info_json_path, 'w') as f:
                json.dump([], f, indent=2)
        
        for file in files:
            file_path = os.path.join(conv_dir, file)
            if not os.path.exists(file_path):
                if file.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump({
                            "face_id": face_id,
                            "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                            "last_updated": time.strftime('%Y-%m-%d %H:%M:%S'),
                            "chat_history_files": [],
                            "knowledge_base": {},
                            "personal_info": []
                        }, f, indent=2)
        
        kb_path = os.path.join(conv_dir, "knowledge_base.txt")
        if not os.path.exists(kb_path):
            with open(kb_path, 'w') as f:
                f.write(f"# Knowledge Base for Face ID: {face_id}\n")
                f.write(f"# Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        return conv_dir
    except Exception as e:
        print(f"Error initializing conversation directory: {e}")
        return None

def update_conversation_files(state, previous_state=None):
    """
    Update the conversation files with the current state.
    This includes the timestamped chat history, knowledge base, and personal info.
    """
    global CURRENT_FACE_ID, CURRENT_RUN_TIMESTAMP
    
    if not CURRENT_FACE_ID or not CURRENT_RUN_TIMESTAMP:
        print("No face ID or run timestamp set, cannot update conversation files")
        return
    
    try:
        conv_dir = os.path.join(os.getcwd(), "conversations", f"conversation_{CURRENT_FACE_ID}")
        if not os.path.exists(conv_dir):
            print(f"Conversation directory does not exist, creating: {conv_dir}")
            init_conversation_directory(CURRENT_FACE_ID)
        
        chat_history_path = os.path.join(conv_dir, f"chat_history_{CURRENT_RUN_TIMESTAMP}.json")
        
        chat_history_updated = False
        
        if "conversation" in state and state["conversation"]:
            conversation_content = state["conversation"]
            
            conversation_turns = []
            
            if conversation_content:
                lines = conversation_content.split('\n')
                current_speaker = None
                current_message = []
                
                for line in lines:
                    if line.startswith('[USER]:'):
                        if current_speaker:
                            conversation_turns.append({
                                "speaker": current_speaker,
                                "message": '\n'.join(current_message).strip()
                            })
                            current_message = []
                        
                        current_speaker = "USER"
                        content = line[len('[USER]:'):].strip()
                        if content: 
                            current_message.append(content)
                    
                    elif line.startswith('[ASSISTANT]:'):
                        if current_speaker:
                            conversation_turns.append({
                                "speaker": current_speaker,
                                "message": '\n'.join(current_message).strip()
                            })
                            current_message = []
                        
                        current_speaker = "ASSISTANT"
                        content = line[len('[ASSISTANT]:'):].strip()
                        if content:  
                            current_message.append(content)
                    
                    elif line.startswith('[SPEAKER '):
                        if current_speaker:
                            conversation_turns.append({
                                "speaker": current_speaker,
                                "message": '\n'.join(current_message).strip()
                            })
                            current_message = []
                        
                        import re
                        speaker_match = re.match(r'\[SPEAKER (\d+)\]:', line)
                        if speaker_match:
                            speaker_num = speaker_match.group(1)
                            current_speaker = f"SPEAKER {speaker_num}"
                            content = line[line.find(':')+1:].strip()
                            if content:  
                                current_message.append(content)
                    
                    elif current_speaker:
                        current_message.append(line)
                
                if current_speaker and current_message:
                    conversation_turns.append({
                        "speaker": current_speaker,
                        "message": '\n'.join(current_message).strip()
                    })
            
            with open(chat_history_path, 'w') as f:
                json.dump(conversation_turns, f, indent=2)
            
            chat_history_updated = True
            
            if chat_history_updated and CURRENT_FACE_ID and CURRENT_RUN_TIMESTAMP:
                face_watcher_instance = None
                if 'face_watcher' in globals():
                    face_watcher_instance = globals()['face_watcher']
                
                if face_watcher_instance:
                    # Use the FaceDirectoryWatcher method that handles run timestamps
                    upload_thread = threading.Thread(
                        target=face_watcher_instance._sync_chat_history_files,
                        args=(CURRENT_FACE_ID, CURRENT_RUN_TIMESTAMP)
                    )
                    upload_thread.daemon = True
                    upload_thread.start()
                    print(f"Started upload of chat history to Supabase for run: {CURRENT_RUN_TIMESTAMP}")
                else:
                    # Fall back to direct update if face_watcher isn't available
                    upload_thread = threading.Thread(
                        target=update_chat_history_in_supabase,
                        args=(CURRENT_FACE_ID, chat_history_path)
                    )
                    upload_thread.daemon = True
                    upload_thread.start()
                    print(f"Started direct upload of chat history to Supabase in background thread")
        
        # KNOWLEDGE BASE: Append new knowledge to knowledge_base.txt (real-time)
        if "knowledge_base" in state and state["knowledge_base"]:
            current_kb = state.get("knowledge_base", {})
            
            # Compare with previous state to find only new knowledge
            prev_kb = {}
            if previous_state and "knowledge_base" in previous_state:
                prev_kb = previous_state.get("knowledge_base", {})
            
            # Calculate new or updated topics
            new_topics = []
            for topic, info in current_kb.items():
                if topic not in prev_kb:
                    new_topics.append((topic, info, "new"))
                elif prev_kb[topic] != info:
                    new_topics.append((topic, info, "updated"))
            
            # If there are new or updated topics, append them to the file
            if new_topics:
                kb_path = os.path.join(conv_dir, "knowledge_base.txt")
                
                # Read existing content to avoid duplication
                existing_content = ""
                if os.path.exists(kb_path):
                    with open(kb_path, 'r') as f:
                        existing_content = f.read()
                
                # Append only new content
                with open(kb_path, 'a') as f:
                    # Add a timestamp for this update if file already has content
                    if existing_content and not existing_content.endswith("\n\n"):
                        f.write("\n\n")
                    f.write(f"# Update: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
                    for topic, info, status in new_topics:
                        # Check if this exact topic is already in the file to avoid duplication
                        topic_header = f"## {topic}"
                        if topic_header in existing_content:
                            # If topic exists but content is different, add as update
                            f.write(f"### Updated: {topic}\n\n")
                        else:
                            f.write(f"## {topic}\n\n")
                        
                        if isinstance(info, list):
                            for item in info:
                                f.write(f"{item}\n\n")
                        elif isinstance(info, dict):
                            for key, value in info.items():
                                f.write(f"- {key}: {value}\n")
                        else:
                            f.write(f"{info}\n")
                        f.write("\n")
                
                print(f"Appended {len(new_topics)} new/updated topics to knowledge base")
                
                # IMPORTANT: When knowledge base is updated due to search, also add relevant data to personal info
                # This ensures search results automatically flow into personal information
                if new_topics and "personal_info" not in state:
                    # If we have new knowledge but no personal info yet, create it
                    state["personal_info"] = []
                
                # Extract personal information from knowledge base updates
                for topic, info, status in new_topics:
                    # Only add new topics (not updates) to personal info
                    if status == "new":
                        # Create a personal info entry from this topic
                        # Use a more robust structure depending on the info type
                        if isinstance(info, dict):
                            for key, value in info.items():
                                if not isinstance(value, (dict, list)):  # Only add simple values
                                    personal_info_item = {
                                        "type": f"{topic}_{key}",
                                        "value": str(value),
                                        "source": "knowledge_base",
                                        "confidence": "high"
                                    }
                                    # Append to state if not already there
                                    if "personal_info" in state:
                                        if isinstance(state["personal_info"], list):
                                            # Check if this info is already in the list
                                            if not any(
                                                item.get("type") == personal_info_item["type"] and 
                                                item.get("value") == personal_info_item["value"]
                                                for item in state["personal_info"] if isinstance(item, dict)
                                            ):
                                                state["personal_info"].append(personal_info_item)
                                                print(f"Added search result to personal info: {personal_info_item['type']}")
        
        # PERSONAL INFO: Update personal info as JSON with append-only updates (real-time)
        if "personal_info" in state:
            # Path to personal info JSON file
            personal_info_json_path = os.path.join(conv_dir, "personal_info.json")
            existing_personal_info = []
            
            # Try to load existing personal info from JSON if it exists
            if os.path.exists(personal_info_json_path):
                try:
                    with open(personal_info_json_path, 'r') as f:
                        existing_personal_info = json.load(f)
                except json.JSONDecodeError:
                    print("Error reading personal info JSON, starting fresh")
                    existing_personal_info = []
            
            # Get current personal info from state
            current_personal_info = state.get("personal_info", [])
            
            # Check if there's anything new to add
            info_changed = False
            new_items_added = []
            
            # IMPROVED DUPLICATE DETECTION
            # Function to check if an item already exists in the list
            def item_exists(new_item, existing_items):
                """Check if an item already exists using multiple comparison strategies"""
                if isinstance(new_item, dict):
                    # Strategy 1: Check by "type" and "value" fields
                    for existing in existing_items:
                        if isinstance(existing, dict):
                            # Case 1: Both type and value match
                            if (new_item.get("type") == existing.get("type") and 
                                new_item.get("value") == existing.get("value")):
                                return True
                            
                            # Case 2: Types match but handle partial value matches
                            if new_item.get("type") == existing.get("type"):
                                new_val = str(new_item.get("value", "")).strip().lower()
                                existing_val = str(existing.get("value", "")).strip().lower()
                                # Check if one contains the other
                                if new_val in existing_val or existing_val in new_val:
                                    print(f"Similar value detected: '{new_val}' vs '{existing_val}'")
                                    return True
                            
                            # Case 3: Values match but types are related
                            if new_item.get("value") == existing.get("value"):
                                new_type = str(new_item.get("type", "")).strip().lower()
                                existing_type = str(existing.get("type", "")).strip().lower()
                                # Check for related types
                                if new_type in existing_type or existing_type in new_type:
                                    print(f"Related types detected: '{new_type}' vs '{existing_type}'")
                                    return True
                                
                    return False
                else:
                    # For non-dict items, simple equality check
                    return new_item in existing_items
            
            # Personal info could be a list or dict, handle both
            if isinstance(current_personal_info, list):
                # For list-based personal info, add new items
                for item in current_personal_info:
                    # Check if this item is already in the existing info
                    if not item_exists(item, existing_personal_info):
                        # Add timestamp to track when this was added
                        if isinstance(item, dict):
                            item_with_timestamp = item.copy()
                            if "timestamp" not in item_with_timestamp:
                                item_with_timestamp["timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S')
                            existing_personal_info.append(item_with_timestamp)
                            item_desc = item.get("type", "Unknown")
                            print(f"Added new personal info: {item_desc}")
                            new_items_added.append(item_desc)
                        else:
                            # Simple values
                            existing_personal_info.append(item)
                            print("Added new simple personal info item")
                            new_items_added.append(str(item))
                        info_changed = True
                    else:
                        print(f"Skipped duplicate personal info: {item.get('type', str(item)[:20]+'...') if isinstance(item, dict) else str(item)}")
            
            elif isinstance(current_personal_info, dict):
                # For dictionary-based personal info
                for key, value in current_personal_info.items():
                    # Convert to list format for consistency
                    new_item = {
                        "type": key,
                        "value": value,
                        "confidence": "high",
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Check if this item is already in the existing info
                    if not item_exists(new_item, existing_personal_info):
                        existing_personal_info.append(new_item)
                        print(f"Added new personal info category: {key}")
                        new_items_added.append(key)
                        info_changed = True
                    else:
                        print(f"Skipped duplicate personal info: {key}")
            
            # Only write to the file if something changed
            if info_changed:
                # Save the updated personal info to JSON
                with open(personal_info_json_path, 'w') as f:
                    json.dump(existing_personal_info, f, indent=2)
                
                print(f"Updated personal info JSON with {len(existing_personal_info)} items")
                if new_items_added:
                    print(f"Newly added items: {', '.join(new_items_added)}")
                
                # Also update the personal info in Supabase
                if CURRENT_FACE_ID:
                    update_thread = threading.Thread(
                        target=update_personal_info_in_supabase,
                        args=(CURRENT_FACE_ID, existing_personal_info)
                    )
                    update_thread.daemon = True
                    update_thread.start()
                    print(f"Started update of personal info in Supabase in background thread")
            else:
                print("No changes to personal info - skipping file update")
        
        # Also update the summary and topics in separate files if needed
        if "summary" in state:
            summary_path = os.path.join(conv_dir, "summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"# Conversation Summary for Face ID: {CURRENT_FACE_ID}\n")
                f.write(f"# Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(state["summary"])
        
        if "topics" in state:
            topics_path = os.path.join(conv_dir, "topics.txt")
            with open(topics_path, 'w') as f:
                f.write(f"# Conversation Topics for Face ID: {CURRENT_FACE_ID}\n")
                f.write(f"# Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for topic in state["topics"]:
                    if isinstance(topic, dict):
                        name = topic.get("name", "Unknown")
                        category = topic.get("category", "")
                        description = topic.get("description", "")
                        
                        f.write(f"## {name}\n")
                        f.write(f"- Category: {category}\n")
                        f.write(f"- Description: {description}\n\n")
                    else:
                        f.write(f"- {topic}\n\n")
        
        # Update the main conversation_data.json with a list of available chat history files
        json_path = os.path.join(conv_dir, "conversation_data.json")
        
        # Try to read existing data
        json_data = {
            "face_id": CURRENT_FACE_ID,
            "last_updated": time.strftime('%Y-%m-%d %H:%M:%S'),
            "chat_history_files": [],
            "knowledge_base": state.get("knowledge_base", {}),
            "personal_info": state.get("personal_info", [])
        }
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                try:
                    existing_data = json.load(f)
                    json_data.update(existing_data)
                except json.JSONDecodeError:
                    pass
        
        # Get list of chat history files
        chat_history_files = []
        for filename in os.listdir(conv_dir):
            if filename.startswith("chat_history_") and filename.endswith(".json"):
                chat_history_files.append(filename)
        
        json_data["chat_history_files"] = chat_history_files
        json_data["current_run"] = CURRENT_RUN_TIMESTAMP
        
        # Write the updated JSON data
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Updated main conversation data file with list of {len(chat_history_files)} chat history files")
    except Exception as e:
        print(f"Error updating conversation files: {e}")
        traceback.print_exc()

def cleanup_temp_files():
    """
    Clean up the temp_files directory by removing all temporary files.
    Preserves the directory structure but deletes all contents.
    """
    try:
        temp_dir = os.path.join(os.getcwd(), "temp_files")
        if os.path.exists(temp_dir):
            print("\n===== CLEANING UP TEMPORARY FILES =====")
            
            # Count files before cleanup
            file_count = 0
            for root, dirs, files in os.walk(temp_dir):
                file_count += len(files)
            
            # Clean files but keep directories
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")
            
            print(f"Cleaned up {file_count} temporary files from {temp_dir}")
            
            # Make sure debug directory still exists
            debug_dir = os.path.join(temp_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
    except Exception as e:
        print(f"Error during temp file cleanup: {e}")

# Monkey patch the ConversationAssistant to update conversation files
original_run_method = None

def patch_run_method(assistant):
    """
    Patch the run method of the ConversationAssistant to update conversation files
    after each conversation turn.
    
    Args:
        assistant: The ConversationAssistant instance to patch
    """
    global original_run_method, last_state
    
    # Save the original method if we haven't already
    if original_run_method is None:
        original_run_method = assistant.run
    
    # Create a wrapper function that calls the original method and then updates files
    def wrapper(*args, **kwargs):
        global last_state
        
        # Call the original method to process the conversation with any args
        # We need to bind the method to the assistant instance
        result = original_run_method.__get__(assistant, type(assistant))(*args, **kwargs)
        
        # Then update conversation files with the latest state if we have a workflow
        if hasattr(assistant, 'workflow') and assistant.workflow and hasattr(assistant.workflow, 'state'):
            # Update with both current and previous state for incremental updates
            current_state = assistant.workflow.state
            update_conversation_files(current_state, last_state)
            last_state = current_state.copy() if current_state else None  # Store a copy of the current state
        
        return result
    
    # Also hook into the state update methods for real-time updates
    if hasattr(assistant, 'workflow') and hasattr(assistant.workflow, 'update_conversation'):
        original_update = assistant.workflow.update_conversation
        
        def update_wrapper(user_input):
            global last_state
            
            # Save current state before update
            if hasattr(assistant.workflow, 'state'):
                prev_state = assistant.workflow.state.copy() if assistant.workflow.state else None
            else:
                prev_state = None
            
            # Call original update method
            result = original_update(user_input)
            
            # Update files after state change
            if hasattr(assistant.workflow, 'state'):
                update_conversation_files(assistant.workflow.state, prev_state)
                last_state = assistant.workflow.state.copy() if assistant.workflow.state else None
            
            return result
        
        # Replace the update method
        assistant.workflow.update_conversation = update_wrapper
    
    # Patch additional LangGraph mechanisms to catch all state changes
    try:
        # Try to patch the LangGraph graph object directly
        if hasattr(assistant, 'workflow') and hasattr(assistant.workflow, 'graph'):
            print("Patching LangGraph workflow for real-time state monitoring...")
            
            # Try to patch any run_node method which is where state changes often happen
            if hasattr(assistant.workflow.graph, 'run_node'):
                original_run_node = assistant.workflow.graph.run_node
                
                def run_node_wrapper(*args, **kwargs):
                    global last_state
                    
                    result = original_run_node(*args, **kwargs)
                    
                    if hasattr(assistant.workflow, 'state'):
                        current_state = assistant.workflow.state
                        
                        if current_state != last_state and 'conversation' in current_state:
                            prev_conv = last_state.get('conversation', '') if last_state else ''
                            current_conv = current_state.get('conversation', '')
                            
                            if current_conv != prev_conv:
                                node_name = args[0] if args else kwargs.get('node', 'unknown')
                                print(f"Conversation state updated after node {node_name}")
                                update_conversation_files(current_state, last_state)
                                last_state = current_state.copy()
                    
                    return result
                
                # Replace the run_node method
                assistant.workflow.graph.run_node = run_node_wrapper
                print("Successfully patched LangGraph run_node method")
            
            # Try to patch the StateManager if it exists
            if hasattr(assistant.workflow.graph, '_state'):
                original_state_setter = assistant.workflow.graph._state.__setattr__
                
                def state_setter_wrapper(name, value):
                    global last_state
                    # Call original setter
                    original_state_setter(name, value)
                    
                    # After state is updated, check if it's a complete state change
                    if name == "state" and hasattr(assistant.workflow, 'state'):
                        new_state = assistant.workflow.state
                        if new_state != last_state and 'conversation' in new_state:
                            prev_conv = last_state.get('conversation', '') if last_state else ''
                            current_conv = new_state.get('conversation', '')
                            
                            # Only update if the conversation changed
                            if current_conv != prev_conv:
                                print("State change detected in LangGraph workflow")
                                update_conversation_files(new_state, last_state)
                                last_state = new_state.copy()
                
                # Replace the state setter
                assistant.workflow.graph._state.__setattr__ = state_setter_wrapper
                print("Successfully patched LangGraph state manager")
                
    except Exception as e:
        print(f"Note: Could not patch deep LangGraph internals: {e}")
        print("State changes will be monitored at conversation update level only.")
    
    # Replace the original run method with our wrapper
    assistant.run = wrapper

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
        
        # Ensure the directory exists
        os.makedirs(self.watch_dir, exist_ok=True)
        
        # Initial scan for existing files
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
        print("Started face directory watcher thread")
        
        # Also start a worker thread to process the queue
        worker_thread = threading.Thread(target=self._process_queue)
        worker_thread.daemon = True
        worker_thread.start()
        print("Started face upload worker thread")
        
        # Check for any files that might have been added while the system was offline
        self._process_unprocessed_files()
    
    def _process_unprocessed_files(self):
        """Check for any unprocessed files in the user-history database and sync chat history"""
        if not supabase_client:
            print("Supabase client not initialized. Cannot check for unprocessed files.")
            return
        
        try:
            print("Checking for unprocessed face images...")
            
            # Get all files in the watch directory
            face_files = {}
            for file_path in glob.glob(os.path.join(self.watch_dir, "face_*.jpg")):
                filename = os.path.basename(file_path)
                user_id = filename[5:-4]  # Remove "face_" prefix and ".jpg" suffix
                face_files[user_id] = file_path
            
            if not face_files:
                print("No face files found in the directory")
                return
            
            # Get all user IDs from the user-history table
            try:
                # Get all existing user IDs with a profile_pic field that's not null
                existing_users = (
                    supabase_client.table("user-history")
                    .select("user_id, profile_pic")
                    .execute()
                )
                
                existing_user_dict = {}
                for user in existing_users.data:
                    existing_user_dict[user["user_id"]] = user.get("profile_pic", "")
                
                print(f"Found {len(existing_user_dict)} existing users in the database")
                
                # Find files that don't have corresponding entries, or have entries without profile pics
                for user_id, file_path in face_files.items():
                    if user_id not in existing_user_dict:
                        print(f"Found unprocessed face image for user ID: {user_id}")
                        self.file_queue.put(file_path)
                    elif not existing_user_dict[user_id]:
                        # User exists but has no profile pic
                        print(f"User ID {user_id} exists but has no profile pic, queueing for update")
                        self.file_queue.put(file_path)
                    else:
                        # User exists and has a profile pic
                        expected_path = f"faces/face_{user_id}.jpg"
                        if expected_path not in existing_user_dict[user_id]:
                            print(f"User ID {user_id} has different profile pic, queueing for update")
                            self.file_queue.put(file_path)
                        else:
                            print(f"Skipping user ID {user_id}, already has correct profile pic")
                    
                    # Sync chat history files for this user ID
                    self._sync_chat_history_files(user_id)
                
            except Exception as e:
                print(f"Error checking existing users: {e}")
                traceback.print_exc()
                
                # If error checking, be more cautious - check if we should queue any files
                print("Using fallback approach: checking each user ID individually")
                for user_id, file_path in face_files.items():
                    try:
                        # Try to check just this one user ID
                        user_check = (
                            supabase_client.table("user-history")
                            .select("user_id, profile_pic")
                            .eq("user_id", user_id)
                            .execute()
                        )
                        
                        if not user_check.data:
                            print(f"User ID {user_id} not found, queueing for processing")
                            self.file_queue.put(file_path)
                        elif not user_check.data[0].get("profile_pic"):
                            print(f"User ID {user_id} has no profile pic, queueing for update")
                            self.file_queue.put(file_path)
                    except:
                        # If we can't even check individual users, don't queue to avoid duplicates
                        print(f"Skipping check for user ID {user_id} due to database errors")
        except Exception as e:
            print(f"Error processing unprocessed files: {e}")
            traceback.print_exc()
    
    def _sync_chat_history_files(self, user_id, specific_timestamp=None):
        """
        Sync chat history files for a specific user ID to the Supabase chat-history table.
        If specific_timestamp is provided, find and upload that specific run's file.
        Otherwise, find and upload the most recent chat history file.
        
        For each run (identified by timestamp), we maintain exactly one entry in the
        Supabase database, updating it as the conversation progresses.
        
        Args:
            user_id: The numeric ID of the user
            specific_timestamp: Optional timestamp to identify a specific chat history file
        """
        # Check if the conversation directory exists
        conv_dir = os.path.join(os.getcwd(), "conversations", f"conversation_{user_id}")
        if not os.path.exists(conv_dir):
            print(f"No conversation directory found for user ID: {user_id}")
            return
        
        # Find relevant chat history files for this user
        chat_history_files = []
        
        # If specific timestamp is provided, find that exact file
        if specific_timestamp:
            target_file = os.path.join(conv_dir, f"chat_history_{specific_timestamp}.json")
            if os.path.exists(target_file):
                chat_history_files.append(target_file)
                print(f"Found specific chat history file for run: {specific_timestamp}")
            else:
                # Try to find files that might contain this timestamp
                for filename in os.listdir(conv_dir):
                    if filename.startswith('chat_history_') and filename.endswith('.json'):
                        if specific_timestamp in filename:
                            chat_history_files.append(os.path.join(conv_dir, filename))
            
            if chat_history_files:
                print(f"Found {len(chat_history_files)} files containing timestamp: {specific_timestamp}")
        
        # If no specific timestamp or no matching files found, get all chat history files
        if not chat_history_files:
            for filename in os.listdir(conv_dir):
                if filename.startswith('chat_history_') and filename.endswith('.json'):
                    chat_history_files.append(os.path.join(conv_dir, filename))
        
        if not chat_history_files:
            print(f"No chat history files found for user ID: {user_id}")
            return
        
        # Determine which file to use
        if specific_timestamp and len(chat_history_files) == 1:
            # If we have exactly one file for the specified timestamp, use it
            target_file = chat_history_files[0]
            print(f"Uploading specified chat history file for run: {specific_timestamp}")
        else:
            # Otherwise, sort by modification time and use the most recent
            chat_history_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
            target_file = chat_history_files[0]
            
            # Extract timestamp from the filename for logging
            file_basename = os.path.basename(target_file)
            file_timestamp = "unknown"
            if file_basename.startswith("chat_history_") and file_basename.endswith(".json"):
                file_timestamp = file_basename[13:-5]
            
            print(f"Uploading chat history file: {file_basename} (timestamp: {file_timestamp})")
        
        # Create a thread to handle the upload
        def upload_thread():
            try:
                # Pass the target file to the update function, which will handle
                # finding/updating the corresponding entry in the database
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
        print(f"Watching directory for new face images: {self.watch_dir}")
        while self.running:
            try:
                # Scan the directory for new files
                for file_path in glob.glob(os.path.join(self.watch_dir, "face_*.jpg")):
                    filename = os.path.basename(file_path)
                    if filename not in self.processed_files:
                        print(f"Found new face image: {filename}")
                        self.file_queue.put(file_path)
                        self.processed_files.add(filename)
            except Exception as e:
                print(f"Error in directory watcher: {e}")
            
            # Sleep before checking again
            time.sleep(2.0)

    def _process_queue(self):
        """Process the queue of files to upload"""
        while self.running:
            try:
                # Get a file from the queue with timeout
                try:
                    file_path = self.file_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the file
                self._process_file(file_path)
                
                # Mark the task as done
                self.file_queue.task_done()
            except Exception as e:
                print(f"Error in queue processor: {e}")

    def _process_file(self, file_path):
        """Process a single file"""
        try:
            filename = os.path.basename(file_path)
            
            # Extract the user ID from the filename (format: face_TIMESTAMP.jpg)
            if filename.startswith("face_") and filename.endswith(".jpg"):
                user_id = filename[5:-4]  # Remove "face_" prefix and ".jpg" suffix
                print(f"Processing new face image for user ID: {user_id}")
                
                # Upload to Supabase
                upload_face_to_supabase(file_path, user_id)
            else:
                print(f"Skipping file with invalid format: {filename}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    try:
        args = parse_arguments()
        
        # Configure environment based on debug flag
        if args.debug:
            import logging
            logging.basicConfig(level=logging.DEBUG)
            print("Debug logging enabled")
            
            # Also enable more detailed exception reporting
            def excepthook(exc_type, exc_value, exc_traceback):
                print("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
            sys.excepthook = excepthook
        
        print(f"Starting Conversation Assistant...")
        
        # Clean up duplicate entries in Supabase
        if supabase_client:
            clean_supabase_duplicates()
        
        # Start the face directory watcher
        face_watcher = FaceDirectoryWatcher()
        face_watcher.start()
            
        if args.screen:
            print(f"Face will be rechecked every {args.face_recheck} seconds")
        
        # Initialize and run the assistant
        assistant = ConversationAssistant(
            use_diarization=args.diarization,
            expected_speakers=args.speakers,
            use_camera=args.screen
        )
        
        patch_run_method(assistant)
        
        if args.screen and hasattr(assistant, 'facial_recognition') and assistant.facial_recognition:
            assistant.facial_recognition.set_recheck_interval(args.face_recheck)
            
            face_id = detect_and_recognize_face(assistant)
            if face_id:
                print(f"Face recognition completed successfully. Identified as: {face_id}")
            
        try:
            assistant.run()
        finally:
            if 'face_watcher' in locals():
                face_watcher.stop()
            
            if CURRENT_FACE_ID and hasattr(assistant, 'workflow') and assistant.workflow:
                update_conversation_files(assistant.workflow.state)
            
            if not args.keep_temp:
                cleanup_temp_files()
        
    except KeyboardInterrupt:
        if 'face_watcher' in locals():
            face_watcher.stop()
            
        if 'assistant' in locals() and CURRENT_FACE_ID and hasattr(assistant, 'workflow') and assistant.workflow:
            update_conversation_files(assistant.workflow.state)
        
        if 'args' in locals() and not args.keep_temp:
            cleanup_temp_files()
    except Exception as e:
        print(f"Startup error: {e}")
        if 'args' in locals() and args.debug:
            traceback.print_exc()
        if 'args' in locals() and not args.keep_temp:
            cleanup_temp_files() 