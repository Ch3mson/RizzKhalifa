#!/usr/bin/env python3

from typing import Dict, List, Tuple, Any, Optional
import os
import tempfile
from groq import Groq
from modules.config import SAMPLE_RATE, CHANNELS, GROQ_MODEL, TRIGGER_PHRASE, STOP_PHRASE
from modules.agents import SpeakerDiarizationAgent

class SpeechToText:
    """
    Uses Groq API to transcribe audio to text.
    Can also perform speaker diarization if enabled.
    """
    def __init__(self, model_name="whisper-large-v3", use_diarization=False):
        print(f"Initializing Groq speech-to-text with model '{model_name}'...")
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model_name
        
        # Speaker diarization
        self.use_diarization = use_diarization
        self.diarization_agent = None
        self.user_reference_path = None
        
        if use_diarization:
            print("Initializing speaker diarization...")
            self.diarization_agent = SpeakerDiarizationAgent()
    
    def set_user_reference(self, reference_path: str) -> bool:
        """Set the user's voice reference for speaker diarization"""
        if not self.use_diarization or self.diarization_agent is None:
            print("Speaker diarization is not enabled")
            return False
            
        result = self.diarization_agent.capture_user_reference(reference_path)
        if result:
            self.user_reference_path = reference_path
            return True
        return False
    
    def _convert_audio_to_base64(self, audio_file: str) -> str:
        """Convert audio file to base64 for API submission"""
        import base64
        with open(audio_file, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def transcribe(self, audio_file: str, detect_trigger_only=False) -> str:
        """
        Transcribe audio to text without speaker diarization.
        
        Args:
            audio_file: Path to the audio file
            detect_trigger_only: If True, optimize for speed to just detect trigger phrases
            
        Returns:
            str: Transcribed text
        """
        try:
            import requests
            import json
            
            # For OpenAI-compatible Whisper API
            with open(audio_file, "rb") as audio:
                # Direct API call to Groq's audio transcription endpoint
                headers = {
                    "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}",
                }
                
                files = {
                    "file": audio,
                }
                
                data = {
                    "model": self.model,
                }
                
                if detect_trigger_only:
                    # Add a prompt parameter for trigger detection
                    data["prompt"] = f"Focus on detecting the phrase '{TRIGGER_PHRASE}' or '{STOP_PHRASE}' if present."
                    # Use temperature=0 for more accurate transcription of specific phrases
                    data["temperature"] = 0
                
                response = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data
                )
                
                if response.status_code != 200:
                    print(f"Error in Groq transcription: Status {response.status_code} - {response.text}")
                    return ""
                    
                result = response.json()
                return result.get("text", "")
        
        except Exception as e:
            print(f"Error in Groq transcription: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def transcribe_with_speakers(self, audio_file: str, num_speakers: int = 2) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Transcribe audio to text with speaker diarization.
        """
        if not self.use_diarization or self.diarization_agent is None:
            print("Speaker diarization is not enabled, falling back to regular transcription")
            return self.transcribe(audio_file), []
        
        # First get the basic transcription
        transcript = self.transcribe(audio_file)
        
        if not transcript:
            return "", []
        
        # Then apply speaker diarization
        # This is a simplified approach - in a real implementation, you'd need to 
        # align the transcript with the audio timing for proper diarization
        segments = [{"text": transcript, "start": 0.0, "end": 10.0}]  # Placeholder timing
        
        segments_with_speakers = self.diarization_agent.process_conversation(
            audio_file, 
            segments, 
            num_speakers=num_speakers
        )
        
        # Create a full transcript with speaker labels
        formatted_segments = []
        current_speaker = None
        for segment in segments_with_speakers:
            if segment["speaker"] != current_speaker:
                current_speaker = segment["speaker"]
                formatted_segments.append(f"\n[{current_speaker}]: {segment['text']}")
            else:
                formatted_segments.append(segment["text"])
        
        full_transcript = " ".join(formatted_segments)
        
        return full_transcript, segments_with_speakers 