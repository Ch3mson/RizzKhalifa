#!/usr/bin/env python3

import os
import torch
from dotenv import load_dotenv
import time

load_dotenv()

# Check if required API keys are set
required_env_vars = ["OPENAI_API_KEY", "SEARCHAPI_API_KEY", "GROQ_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    print("Please set these in your .env file")
    exit(1)

SAMPLE_RATE = 16000
CHANNELS = 1
WHISPER_MODEL = "small"
OPENAI_MODEL = "gpt-4o-mini"
GROQ_MODEL= "llama-3.3-70b-versatile"
GROQ_WHISPER_MODEL = "whisper-large-v3"
BUFFER_SIZE = 512
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.0
CONVERSATION_PAUSE = 2.0
CONTINUOUS_PROCESSING_INTERVAL = 5.0

USE_GPU = torch.cuda.is_available()
COMPUTE_TYPE = "float16" if USE_GPU else "int8"
DEVICE = "cuda" if USE_GPU else "cpu"

TRIGGER_PHRASE = "let me think"
STOP_PHRASE = "ok ok ok"

# Directory for storing conversation logs
CONVERSATIONS_DIR = "conversations"

# Function to generate a timestamped filename for conversation logs
def get_output_file():
    """Get the path to save the conversation file."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create logs directory inside conversations folder
    logs_dir = os.path.join(os.getcwd(), "conversations", "system_logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    return os.path.join(logs_dir, f"conversation-{timestamp}.txt")

# For backward compatibility (will be initialized at runtime)
OUTPUT_FILE = None

from typing import List, Dict, Any, TypedDict

class ConversationState(TypedDict):
    conversation: str
    summary: str
    topics: List[Dict[str, str]]
    knowledge_base: Dict[str, List[str]]
    personal_info: List[Dict[str, Any]]
    last_processed: str
    speaker_segments: List[Dict[str, Any]]
    category: str
    restart_count: int
    last_restart: float
    _routing: str

CONVERSATION_STATE_SCHEMA: ConversationState = {
    "conversation": "",
    "summary": "",
    "topics": [],
    "knowledge_base": {},
    "personal_info": [],
    "last_processed": "",
    "speaker_segments": [],
    "category": "",
    "restart_count": 0,
    "last_restart": 0.0,
    "_routing": ""
} 