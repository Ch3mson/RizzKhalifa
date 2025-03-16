#!/usr/bin/env python3

import sys
import argparse
import os
import traceback
import time
import cv2
import shutil
import json
from modules.assistant import ConversationAssistant

# Global variable to store the current face ID
CURRENT_FACE_ID = None
last_state = None
# Add global variable to store the current run timestamp
CURRENT_RUN_TIMESTAMP = None

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

def test_workflow():
    """Test the workflow functionality directly without audio input"""
    from modules.workflow import ConversationWorkflow
    print("\n===== TESTING WORKFLOW FUNCTIONALITY =====")
    
    # Create test directory
    log_dir = os.path.join(os.getcwd(), "test_results")
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize workflow
    workflow = ConversationWorkflow()
    
    # Test inputs that should trigger search
    test_inputs = [
        "Tell me about quantum computing",
        "What's happening with AI developments in 2024?",
        "I'm interested in learning about clean energy innovations"
    ]
    
    results = {}
    
    # Process each test input
    for i, test_input in enumerate(test_inputs):
        print(f"\n\n==== Test {i+1}: {test_input} ====")
        try:
            # Get baseline state
            if hasattr(workflow, 'graph') and workflow.graph:
                print("Using workflow graph to process")
                initial_state = workflow.state.copy()
                workflow.update_conversation(test_input)
                results[f"test_{i+1}"] = {
                    "input": test_input,
                    "success": True,
                    "has_knowledge_base": "knowledge_base" in workflow.state,
                    "kb_size": len(workflow.state.get("knowledge_base", {}))
                }
            else:
                print("⚠ Workflow graph not initialized correctly")
                results[f"test_{i+1}"] = {
                    "input": test_input,
                    "success": False,
                    "error": "Workflow graph not initialized"
                }
        except Exception as e:
            print(f"❌ Test {i+1} failed with error: {e}")
            traceback.print_exc()
            results[f"test_{i+1}"] = {
                "input": test_input,
                "success": False,
                "error": str(e)
            }
    
    # Print results summary
    print("\n===== TEST RESULTS SUMMARY =====")
    success_count = sum(1 for r in results.values() if r["success"])
    print(f"Successful tests: {success_count}/{len(test_inputs)}")
    
    for test_id, result in results.items():
        status = "✓" if result["success"] else "❌"
        kb_info = f", KB: {result['kb_size']} topics" if result["success"] and "kb_size" in result else ""
        error = f" - Error: {result['error']}" if not result["success"] and "error" in result else ""
        print(f"{status} {test_id}: '{result['input'][:30]}...'{kb_info}{error}")
    
    return success_count == len(test_inputs)

def load_existing_conversation_data(face_id):
    """
    Load existing conversation data for a recognized face.
    
    Args:
        face_id: The numeric ID of the detected face
        
    Returns:
        dict: The loaded conversation data or None if not found
    """
    try:
        # Path to the main JSON file
        json_path = os.path.join(os.getcwd(), "conversations", f"conversation_{face_id}", "conversation_data.json")
        
        # Check if the file exists
        if not os.path.exists(json_path):
            print(f"No existing conversation data found for face ID: {face_id}")
            return None
        
        # Load the main data file
        with open(json_path, 'r') as f:
            data = json.load(f)
            print(f"Loaded existing conversation data for face ID: {face_id}")
            print(f"Last conversation updated: {data.get('last_updated', 'unknown')}")
            
            # Get list of available chat history files
            chat_history_files = data.get("chat_history_files", [])
            
            if chat_history_files:
                print(f"Found {len(chat_history_files)} previous conversation sessions")
                
                # Sort by timestamp to get the most recent one
                chat_history_files.sort(reverse=True)
                most_recent = chat_history_files[0]
                
                # Load the most recent chat history - but DON'T use it to initialize the workflow state
                # The conversation history is now just the turns of dialog, not the complete state
                most_recent_path = os.path.join(os.getcwd(), "conversations", f"conversation_{face_id}", most_recent)
                if os.path.exists(most_recent_path):
                    with open(most_recent_path, 'r') as f2:
                        try:
                            most_recent_data = json.load(f2)
                            print(f"Loaded most recent conversation from: {most_recent}")
                            print(f"Found {len(most_recent_data)} conversation turns")
                        except json.JSONDecodeError:
                            print(f"Error reading chat history file: {most_recent}")
            
            # Load knowledge base from existing state
            knowledge_base = data.get("knowledge_base", {})
            if knowledge_base:
                print(f"Loaded knowledge base with {len(knowledge_base)} topics")
            
            # Load personal info from JSON if available
            personal_info_json_path = os.path.join(os.getcwd(), "conversations", f"conversation_{face_id}", "personal_info.json")
            if os.path.exists(personal_info_json_path):
                try:
                    with open(personal_info_json_path, 'r') as f:
                        personal_info_data = json.load(f)
                        data["personal_info"] = personal_info_data
                        print(f"Loaded personal info with {len(personal_info_data)} items")
                except json.JSONDecodeError:
                    print("Error reading personal info JSON file")
            
            # Print a summary
            kb = data.get("knowledge_base", {})
            personal_info = data.get("personal_info", [])
            
            print(f"Found {len(kb)} knowledge base topics")
            print(f"Found {len(personal_info)} personal info items")
            
            return data
    except Exception as e:
        print(f"Error loading conversation data: {e}")
        return None

def detect_and_recognize_face(assistant):
    """
    Captures a single frame, detects a face, and compares it with stored faces.
    If a match is found, returns the face ID.
    If no match, saves the new face for future recognition.
    """
    global CURRENT_FACE_ID, last_state
    
    if not assistant.use_camera or not assistant.facial_recognition:
        print("Camera or facial recognition is not enabled. Skipping face detection.")
        return None
    
    print("\n===== STARTING FACIAL RECOGNITION =====")
    print("Capturing frame for face detection...")
    # Capture a single frame
    frame = assistant._capture_screen_frame()
    if frame is None:
        print("Error: Failed to capture frame for face detection.")
        return None
    
    print(f"Successfully captured frame with shape: {frame.shape}")
    
    # Process the frame with our improved face recognition method
    try:
        print("Calling InsightFace facial recognition...")
        face_id, is_new_face = assistant.facial_recognition.manage_face_recognition(frame)
        
        if face_id:
            # Extract the numeric ID from the face_id (format: "face_TIMESTAMP")
            numeric_id = face_id.split('_')[1]
            CURRENT_FACE_ID = numeric_id
            print(f"Setting current face ID to: {CURRENT_FACE_ID}")
            
            # Initialize conversation directory for this face
            init_conversation_directory(numeric_id)
            
            # For existing faces, try to load previous conversation data
            if not is_new_face:
                print(f"Success: Recognized existing face with ID: {face_id}")
                
                # Load existing conversation data
                existing_data = load_existing_conversation_data(numeric_id)
                
                # If we successfully loaded data and the assistant has a workflow, initialize it
                if existing_data and hasattr(assistant, 'workflow'):
                    # Try to initialize the workflow state with existing data
                    try:
                        print("Attempting to restore previous conversation state...")
                        
                        # Initialize the workflow state if needed
                        if not hasattr(assistant.workflow, 'state') or not assistant.workflow.state:
                            assistant.workflow.state = {}
                        
                        # Copy elements from existing data to workflow state
                        if "chat_history" in existing_data and existing_data["chat_history"]:
                            assistant.workflow.state["chat_history"] = existing_data["chat_history"]
                            print(f"Restored {len(existing_data['chat_history'])} previous messages")
                        
                        if "knowledge_base" in existing_data and existing_data["knowledge_base"]:
                            assistant.workflow.state["knowledge_base"] = existing_data["knowledge_base"]
                            print(f"Restored {len(existing_data['knowledge_base'])} knowledge base topics")
                        
                        if "personal_info" in existing_data and existing_data["personal_info"]:
                            assistant.workflow.state["personal_info"] = existing_data["personal_info"]
                            print(f"Restored {len(existing_data['personal_info'])} personal info categories")
                        
                        # Store the current state as the last state to prevent duplicate updates
                        last_state = assistant.workflow.state.copy() if assistant.workflow.state else None
                        
                    except Exception as e:
                        print(f"Error restoring conversation state: {e}")
                        traceback.print_exc()
            else:
                print(f"Success: New face detected and saved with ID: {face_id}")
            
            # Set the current face for the conversation
            assistant.facial_recognition.update_current_face(
                face_name=face_id,
                face_embedding=None  # The embedding is already saved with the face
            )
            
            return face_id
        else:
            print("Error: No face detected or recognition failed.")
            # Try again with a different approach - take multiple samples
            print("Attempting detection with alternative approach...")
            for _ in range(3):  # Try up to 3 more times
                time.sleep(1)  # Wait a second between attempts
                frame = assistant._capture_screen_frame()
                if frame is not None:
                    try:
                        face_id, is_new_face = assistant.facial_recognition.manage_face_recognition(frame)
                        if face_id:
                            # Extract the numeric ID from the face_id
                            numeric_id = face_id.split('_')[1]
                            CURRENT_FACE_ID = numeric_id
                            print(f"Setting current face ID to: {CURRENT_FACE_ID}")
                            
                            # Initialize conversation directory for this face
                            init_conversation_directory(numeric_id)
                            
                            # For existing faces, try to load previous conversation data
                            if not is_new_face:
                                # Load existing conversation data
                                existing_data = load_existing_conversation_data(numeric_id)
                                
                                # If we successfully loaded data and the assistant has a workflow, initialize it
                                if existing_data and hasattr(assistant, 'workflow'):
                                    # Initialize workflow state with existing data
                                    try:
                                        print("Attempting to restore previous conversation state...")
                                        
                                        # Initialize the workflow state if needed
                                        if not hasattr(assistant.workflow, 'state') or not assistant.workflow.state:
                                            assistant.workflow.state = {}
                                        
                                        # Copy elements from existing data to workflow state
                                        if "chat_history" in existing_data and existing_data["chat_history"]:
                                            assistant.workflow.state["chat_history"] = existing_data["chat_history"]
                                            print(f"Restored {len(existing_data['chat_history'])} previous messages")
                                        
                                        if "knowledge_base" in existing_data and existing_data["knowledge_base"]:
                                            assistant.workflow.state["knowledge_base"] = existing_data["knowledge_base"]
                                            print(f"Restored {len(existing_data['knowledge_base'])} knowledge base topics")
                                        
                                        if "personal_info" in existing_data and existing_data["personal_info"]:
                                            assistant.workflow.state["personal_info"] = existing_data["personal_info"]
                                            print(f"Restored {len(existing_data['personal_info'])} personal info categories")
                                        
                                        # Store the current state as the last state to prevent duplicate updates
                                        last_state = assistant.workflow.state.copy() if assistant.workflow.state else None
                                        
                                    except Exception as e:
                                        print(f"Error restoring conversation state: {e}")
                                        traceback.print_exc()
                            
                            print(f"Success on retry: Face detected with ID: {face_id}")
                            assistant.facial_recognition.update_current_face(
                                face_name=face_id,
                                face_embedding=None
                            )
                            return face_id
                    except Exception as e:
                        print(f"Error during retry: {e}")
            
            print("All retry attempts failed. No face detected.")
            return None
    except Exception as e:
        print(f"Error during face recognition: {e}")
        import traceback
        traceback.print_exc()
        return None

def init_conversation_directory(face_id):
    """
    Initialize the conversation directory structure for a specific face ID.
    Creates a conversations/conversation_{id} directory with necessary files.
    
    Args:
        face_id: The numeric ID of the detected face
    """
    global CURRENT_RUN_TIMESTAMP
    
    try:
        # Generate a timestamp for this run
        CURRENT_RUN_TIMESTAMP = time.strftime('%Y%m%d_%H%M%S')
        
        # Create the conversation directory inside the conversations directory
        conv_dir = os.path.join(os.getcwd(), "conversations", f"conversation_{face_id}")
        os.makedirs(conv_dir, exist_ok=True)
        
        print(f"Initialized conversation directory: {conv_dir}")
        
        # Create the new timestamped chat history file - empty array for conversation turns
        chat_history_path = os.path.join(conv_dir, f"chat_history_{CURRENT_RUN_TIMESTAMP}.json")
        
        # Initialize the JSON file with empty array - just for conversation
        with open(chat_history_path, 'w') as f:
            json.dump([], f, indent=2)
            
        print(f"Created new chat history file for this run: {chat_history_path}")
        
        # Create other necessary files if they don't exist
        files = [
            "conversation_data.json"
        ]
        
        # Initialize personal_info.json if it doesn't exist
        personal_info_json_path = os.path.join(conv_dir, "personal_info.json")
        if not os.path.exists(personal_info_json_path):
            with open(personal_info_json_path, 'w') as f:
                json.dump([], f, indent=2)
            print(f"Created empty personal info JSON file")
        
        for file in files:
            file_path = os.path.join(conv_dir, file)
            if not os.path.exists(file_path):
                # For JSON file, initialize with empty structure
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
        
        # Create the text knowledge base file
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
    
    Args:
        state: The current LangGraph state containing conversation data
        previous_state: The previous state to compare against for incremental updates
    """
    global CURRENT_FACE_ID, CURRENT_RUN_TIMESTAMP
    
    if not CURRENT_FACE_ID or not CURRENT_RUN_TIMESTAMP:
        print("No face ID or run timestamp set, cannot update conversation files")
        return
    
    try:
        # Ensure path is inside conversations directory
        conv_dir = os.path.join(os.getcwd(), "conversations", f"conversation_{CURRENT_FACE_ID}")
        if not os.path.exists(conv_dir):
            print(f"Conversation directory does not exist, creating: {conv_dir}")
            init_conversation_directory(CURRENT_FACE_ID)
        
        # Update timestamped chat history JSON file - containing ONLY the conversation
        chat_history_path = os.path.join(conv_dir, f"chat_history_{CURRENT_RUN_TIMESTAMP}.json")
        
        # Only proceed if we have conversation content
        if "conversation" in state and state["conversation"]:
            # Process conversation content (existing code) 
            conversation_content = state["conversation"]
            
            # Process the raw conversation text into a structured JSON format
            # The conversation usually has format like:
            # [USER]: message content
            # [ASSISTANT]: response content
            
            conversation_turns = []
            
            # Split by speaker turns and process
            if conversation_content:
                lines = conversation_content.split('\n')
                current_speaker = None
                current_message = []
                
                for line in lines:
                    if line.startswith('[USER]:'):
                        # If we were building a previous message, save it
                        if current_speaker:
                            conversation_turns.append({
                                "speaker": current_speaker,
                                "message": '\n'.join(current_message).strip()
                            })
                            current_message = []
                        
                        # Start a new user message
                        current_speaker = "USER"
                        content = line[len('[USER]:'):].strip()
                        if content:  # Only add if there's content on this line
                            current_message.append(content)
                    
                    elif line.startswith('[ASSISTANT]:'):
                        # If we were building a previous message, save it
                        if current_speaker:
                            conversation_turns.append({
                                "speaker": current_speaker,
                                "message": '\n'.join(current_message).strip()
                            })
                            current_message = []
                        
                        # Start a new assistant message
                        current_speaker = "ASSISTANT"
                        content = line[len('[ASSISTANT]:'):].strip()
                        if content:  # Only add if there's content on this line
                            current_message.append(content)
                    
                    elif line.startswith('[SPEAKER '):
                        # If we were building a previous message, save it
                        if current_speaker:
                            conversation_turns.append({
                                "speaker": current_speaker,
                                "message": '\n'.join(current_message).strip()
                            })
                            current_message = []
                        
                        # Extract speaker number and start new message
                        import re
                        speaker_match = re.match(r'\[SPEAKER (\d+)\]:', line)
                        if speaker_match:
                            speaker_num = speaker_match.group(1)
                            current_speaker = f"SPEAKER {speaker_num}"
                            content = line[line.find(':')+1:].strip()
                            if content:  # Only add if there's content on this line
                                current_message.append(content)
                    
                    elif current_speaker:
                        # Continue building the current message
                        current_message.append(line)
                
                # Don't forget to add the last message if there is one
                if current_speaker and current_message:
                    conversation_turns.append({
                        "speaker": current_speaker,
                        "message": '\n'.join(current_message).strip()
                    })
            
            # Write just the conversation turns to the JSON file
            with open(chat_history_path, 'w') as f:
                json.dump(conversation_turns, f, indent=2)
            
            print(f"Updated chat history file with {len(conversation_turns)} conversation turns")
        
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
            last_state = current_state.copy() if current_state else {}  # Store a copy of the current state
        
        return result
    
    # Also hook into the state update methods for real-time updates
    if hasattr(assistant, 'workflow') and hasattr(assistant.workflow, 'update_conversation'):
        original_update = assistant.workflow.update_conversation
        
        def update_wrapper(user_input):
            global last_state
            
            # Save current state before update
            if hasattr(assistant.workflow, 'state'):
                prev_state = assistant.workflow.state.copy() if assistant.workflow.state else {}
            else:
                prev_state = {}
            
            # Call original update method
            result = original_update(user_input)
            
            # Update files after state change
            if hasattr(assistant.workflow, 'state'):
                update_conversation_files(assistant.workflow.state, prev_state)
                last_state = assistant.workflow.state.copy() if assistant.workflow.state else {}
            
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
                    
                    # Call original method
                    result = original_run_node(*args, **kwargs)
                    
                    # After node execution, check if state changed
                    if hasattr(assistant.workflow, 'state'):
                        current_state = assistant.workflow.state
                        
                        # Only update if there's been a meaningful change
                        if current_state != last_state and 'conversation' in current_state:
                            prev_conv = last_state.get('conversation', '') if last_state else ''
                            current_conv = current_state.get('conversation', '')
                            
                            # Check if the conversation content actually changed
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
        
        # Handle test mode
        if args.test:
            success = test_workflow()
            # Clean up temp files if not in debug mode
            if not args.keep_temp:
                cleanup_temp_files()
            sys.exit(0 if success else 1)
        
        # Normal execution
        if args.diarization:
            print(f"Speaker diarization enabled (expected speakers: {args.speakers})")
            print("You'll be asked to provide a 10-second voice sample for identification")
        else:
            print("Speaker diarization disabled")
            
        if args.screen:
            print("Screen capture enabled for facial recognition")
            print("Faces will be automatically detected and matched during conversation")
            print(f"Face will be rechecked every {args.face_recheck} seconds")
        
        # Initialize and run the assistant
        assistant = ConversationAssistant(
            use_diarization=args.diarization,
            expected_speakers=args.speakers,
            use_camera=args.screen
        )
        
        # Monkey patch the run method to update conversation files
        patch_run_method(assistant)
        
        # If facial recognition is enabled, set the recheck interval
        if args.screen and hasattr(assistant, 'facial_recognition') and assistant.facial_recognition:
            assistant.facial_recognition.set_recheck_interval(args.face_recheck)
            
            # Detect and recognize face at startup
            face_id = detect_and_recognize_face(assistant)
            if face_id:
                print(f"Face recognition completed successfully. Identified as: {face_id}")
            
        try:
            assistant.run()
        finally:
            # Final update of conversation files if we have a face ID
            if CURRENT_FACE_ID and hasattr(assistant, 'workflow') and assistant.workflow:
                update_conversation_files(assistant.workflow.state)
            
            # Clean up temp files unless explicitly told to keep them
            if not args.keep_temp:
                cleanup_temp_files()
        
    except KeyboardInterrupt:
        print("\nExiting program due to keyboard interrupt")
        # Final update of conversation files before exit
        if 'assistant' in locals() and CURRENT_FACE_ID and hasattr(assistant, 'workflow') and assistant.workflow:
            update_conversation_files(assistant.workflow.state)
        
        # Clean up temp files unless explicitly told to keep them
        if 'args' in locals() and not args.keep_temp:
            cleanup_temp_files()
    except Exception as e:
        print(f"Startup error: {e}")
        if 'args' in locals() and args.debug:
            traceback.print_exc()
        # Clean up temp files unless explicitly told to keep them
        if 'args' in locals() and not args.keep_temp:
            cleanup_temp_files() 