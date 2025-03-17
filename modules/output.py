#!/usr/bin/env python3

from typing import Dict, Any
from datetime import datetime
import os
from modules.config import CONVERSATIONS_DIR, get_output_file
import json
import time

class TextOutput:
    """Handles text output to console and files"""
    
    def __init__(self):
        if not os.path.exists(CONVERSATIONS_DIR):
            os.makedirs(CONVERSATIONS_DIR)
        
    def output(self, text: str, prefix="ASSISTANT"):
        if not text.strip():
            return
        
    def save_to_file(self, state, filename=None):
        """
        Save the current state to a file.
        """
        try:
            logs_dir = os.path.join(os.getcwd(), "conversations", "system_logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            if filename:
                output_file = os.path.join(logs_dir, filename)
            else:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                output_file = os.path.join(logs_dir, f"workflow_state_{timestamp}.json")
            
            with open(output_file, 'w') as f:
                clean_state = {}
                for key, value in state.items():
                    if key in ['conversation', 'summary', 'knowledge_base', 'topics', 'personal_info']:
                        clean_state[key] = value
                
                json.dump(clean_state, f, indent=2)
                
        except Exception as e:
            print(f"Error saving state to file: {e}")
            import traceback
            traceback.print_exc()

    def _wrap_text(self, text: str, width: int) -> str:
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 > width:
                lines.append(current_line.strip())
                current_line = word
            else:
                current_line += " " + word
        if current_line:
            lines.append(current_line.strip())
        return "\n".join(lines) 