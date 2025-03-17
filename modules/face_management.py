import os
import time
import json
import traceback
import threading
from modules.supabase_integration import upload_face_to_supabase

CURRENT_FACE_ID = None
CURRENT_RUN_TIMESTAMP = None

def detect_and_recognize_face(assistant):
    """
    Captures a single frame, detects a face, and compares it with stored faces.
    """
    global CURRENT_FACE_ID, last_state
    
    if not assistant.use_camera or not assistant.facial_recognition:
        return None
    frame = assistant._capture_screen_frame()
    if frame is None:
        return None
    
    try:
        face_id, is_new_face = assistant.facial_recognition.manage_face_recognition(frame)
        
        if face_id:
            numeric_id = face_id.split('_')[1]
            CURRENT_FACE_ID = numeric_id
            
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
                from modules.conversation_utils import load_existing_conversation_data
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
                        
                        from main import last_state
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
                                from modules.conversation_utils import load_existing_conversation_data
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
                                        
                                        from main import last_state
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