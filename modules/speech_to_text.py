#!/usr/bin/env python3

from typing import Dict, List, Tuple, Any, Optional
from faster_whisper import WhisperModel
from modules.config import WHISPER_MODEL, DEVICE, COMPUTE_TYPE
from modules.agents import SpeakerDiarizationAgent

class SpeechToText:
    """
    Uses Faster-Whisper to transcribe audio to text.
    Can also perform speaker diarization if enabled.
    """
    def __init__(self, model_name=WHISPER_MODEL, use_diarization=False):
        print(f"Loading Whisper model '{model_name}'...")
        self.model = WhisperModel(
            model_name,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root="./models",
        )
        
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
    
    def transcribe(self, audio_file: str) -> str:
        """
        Transcribe audio to text without speaker diarization.
        """
        segments, _ = self.model.transcribe(
            audio_file, 
            beam_size=1,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        return " ".join([segment.text for segment in segments])
    
    def transcribe_with_speakers(self, audio_file: str, num_speakers: int = 2) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Transcribe audio to text with speaker diarization.
        """
        if not self.use_diarization or self.diarization_agent is None:
            print("Speaker diarization is not enabled, falling back to regular transcription")
            return self.transcribe(audio_file), []
        
        # Get segments from Whisper
        result_segments, _ = self.model.transcribe(
            audio_file, 
            beam_size=1,
            language="en",
            vad_filter=True,
            word_timestamps=True,  # Need timestamps for diarization
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Convert to list and add start/end times
        segments = []
        for segment in result_segments:
            segments.append({
                "text": segment.text,
                "start": segment.start,
                "end": segment.end
            })
        
        if not segments:
            return "", []
        
        # Apply speaker diarization
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