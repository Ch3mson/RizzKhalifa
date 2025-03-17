import os
import time
import json
import traceback
import threading
from modules.supabase_integration import update_personal_info_in_supabase, update_chat_history_in_supabase

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

def update_conversation_files(state, previous_state=None):
    """
    Update the conversation files with the current state.
    This includes the timestamped chat history, knowledge base, and personal info.
    """
    from modules.face_management import CURRENT_FACE_ID, CURRENT_RUN_TIMESTAMP
    
    if not CURRENT_FACE_ID or not CURRENT_RUN_TIMESTAMP:
        print("No face ID or run timestamp set, cannot update conversation files")
        return
    
    try:
        conv_dir = os.path.join(os.getcwd(), "conversations", f"conversation_{CURRENT_FACE_ID}")
        if not os.path.exists(conv_dir):
            print(f"Conversation directory does not exist, creating: {conv_dir}")
            from modules.face_management import init_conversation_directory
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

# Original run method to be saved
original_run_method = None

def patch_run_method(assistant):
    """
    Patch the run method of the ConversationAssistant to update conversation files
    after each conversation turn.
    
    Args:
        assistant: The ConversationAssistant instance to patch
    """
    global original_run_method
    from main import last_state
    
    # Save the original method if we haven't already
    if original_run_method is None:
        original_run_method = assistant.run
    
    # Create a wrapper function that calls the original method and then updates files
    def wrapper(*args, **kwargs):
        from main import last_state
        
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
            from main import last_state
            
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
                    from main import last_state
                    
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
                    from main import last_state
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