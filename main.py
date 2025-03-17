#!/usr/bin/env python3

import sys
import os
import traceback
import time
import logging
import threading

# Import modules
from modules.assistant import ConversationAssistant
from modules.utils import parse_arguments, cleanup_temp_files
from modules.supabase_integration import supabase_client, clean_supabase_duplicates
from modules.face_management import detect_and_recognize_face, CURRENT_FACE_ID, CURRENT_RUN_TIMESTAMP
from modules.conversation_utils import update_conversation_files, patch_run_method
from modules.face_watcher import FaceDirectoryWatcher

# For cross-module state management
last_state = None
face_watcher = None

if __name__ == "__main__":
    try:
        args = parse_arguments()
        
        # Configure environment based on debug flag
        if args.debug:
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
            if face_watcher:
                face_watcher.stop()
            
            if CURRENT_FACE_ID and hasattr(assistant, 'workflow') and assistant.workflow:
                update_conversation_files(assistant.workflow.state)
            
            if not args.keep_temp:
                cleanup_temp_files()
        
    except KeyboardInterrupt:
        if face_watcher:
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