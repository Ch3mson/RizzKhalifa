import os
import time
import json
import traceback
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

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
    """
    if not supabase_client:
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