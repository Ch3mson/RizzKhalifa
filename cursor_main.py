#!/usr/bin/env python3
"""
we need to run this simultaneously with the other main.py file.
"""
import os
import time
import queue
import threading
import argparse
import traceback
import sys
import cv2
from datetime import datetime
import wave
import contextlib
import requests
from typing import List, Dict

from modules.config import TRIGGER_PHRASE, STOP_PHRASE, get_output_file, GROQ_WHISPER_MODEL, GROQ_MODEL
from modules.audio import AudioRecorder
from modules.speech_to_text import SpeechToText
from modules.output import TextOutput
from modules.agents.rizz_cursor_agent import RizzCursorAgent
from modules.facial_recognition import FacialRecognitionModule

class CursorAssistant:
    """
    Simplified conversation assistant that:
    1. Uses Whisper model and speech diarization
    2. Saves conversation history
    3. Toggles active listening mode with trigger words
    4. Uses the cursor rizz agent to provide smooth, charismatic dating responses
    """
    def __init__(self, use_diarization=True, expected_speakers=2, use_camera=False):
        """
        Initialize the Cursor Assistant
        """
        self.transcriber = SpeechToText(model_name=GROQ_WHISPER_MODEL, use_diarization=use_diarization)
        self.text_output = TextOutput()
        
        self.recorder = AudioRecorder()
        self.use_diarization = use_diarization
        self.expected_speakers = expected_speakers
        self.user_reference_captured = False
        self.use_camera = use_camera
        self.facial_recognition = None
        self.video_capture = None
        self.video_thread = None
        self.video_recording = False
        self.current_video_buffer = []
        
        self.rizz_agent = RizzCursorAgent()
        
        self.conversations_dir = os.path.join(os.getcwd(), "conversations")
        os.makedirs(self.conversations_dir, exist_ok=True)
        self.processing_queue = queue.Queue()
        self.processing_thread = None
        self.processing_active = False
        
        self.conversation_history = []
        self.speaker_segments = []
        self.is_running = False
        self.active_listening = False
        
        self.recent_transcriptions = []
        self.max_recent_transcriptions = 5
        self.repetition_threshold = 3  
        
        if self.use_camera:
            try:
                self.facial_recognition = FacialRecognitionModule()
                print("Facial recognition module initialized")
            except Exception as e:
                print(f"Error initializing facial recognition: {e}. Proceeding without it.")
                self.facial_recognition = None
                self.use_camera = False

    def process_audio_segment(self, active_mode=False):
        """
        Grabs an audio snippet, transcribes it, checks for trigger phrases.
        Returns (transcription, pause_detected).
        """
        try:
            if not self.active_listening:
                audio_file, _ = self.recorder.get_audio_segment(
                    wait_for_pause=False,
                    trigger_detection=True  
                )
                
                if audio_file:
                    if self._check_for_trigger_phrase(audio_file):
                        os.unlink(audio_file) 
                        audio_file, pause_detected = self.recorder.get_audio_segment(
                            wait_for_pause=active_mode
                        )
                    else:
                        pause_detected = False
                else:
                    audio_file, pause_detected = self.recorder.get_audio_segment(
                        wait_for_pause=active_mode
                    )
            else:
                audio_file, pause_detected = self.recorder.get_audio_segment(
                    wait_for_pause=active_mode
                )
            
            if not audio_file:
                return None, False
            
            if audio_file:
                with contextlib.closing(wave.open(audio_file, 'r')) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
                    
                    if duration < 0.5:
                        os.unlink(audio_file)  
                        return None, False
            
            if self._is_repetitive_transcription(audio_file):
                print(f"Filtering repetitive transcription: {audio_file}")
                os.unlink(audio_file) 
                return None, False
            
            if self.use_diarization and self.user_reference_captured:
                transcription, segments = self.transcriber.transcribe_with_speakers(
                    audio_file, 
                    num_speakers=self.expected_speakers
                )
                
                if not transcription or not transcription.strip():
                    os.unlink(audio_file)  
                    return None, False
                
                print(f"Transcribed with {len(segments)} speaker segments: {transcription[:100]}...")
                
                filtered_segments = []
                for segment in segments:
                    segment_text = segment.get("text", "").strip().lower()
                    
                    if ("thank you" in segment_text or 
                        segment_text.startswith("thank") or 
                        segment_text == "thanks" or 
                        "you're welcome" in segment_text):
                        continue
                        
                    if segment_text == "you" or segment_text in ["you.", "you?", "you!"] or len(segment_text) < 3:
                        print(f"Filtering out noise segment: '{segment_text}'")
                        continue
                        
                    filtered_segments.append(segment)
                
                if not filtered_segments:
                    print("All segments were filtered as noise")
                    os.unlink(audio_file) 
                    return None, False
                
                self.speaker_segments.extend(filtered_segments)
                for segment in filtered_segments:
                    self.conversation_history.append({
                        "timestamp": time.time(),
                        "speaker": segment.get("speaker", "UNKNOWN"),
                        "person": segment.get("person", "UNKNOWN"),
                        "text": segment.get("text", "")
                    })
                
                if segments and self.active_listening:
                    last_segment = segments[-1]
                    current_speaker = last_segment.get("speaker", "UNKNOWN")
                    
                    state = self._get_state_from_history()
                    
                    if self.rizz_agent.is_ready_to_generate(current_speaker, self.active_listening, state=state):
                        print(f"\n🔍 Generating information for you to respond to {current_speaker}...")
                        suggestion = self.rizz_agent.generate_suggestion(
                            state=state,
                            speaker=current_speaker
                        )
                        
                        if suggestion:
                            print(f"\n💡 INFORMATION: {suggestion}\n")
                
                if self.use_camera and self.facial_recognition and segments:
                    video_file = None
                    if self.current_video_buffer:
                        video_file = self._save_temp_video()
                        
                        if video_file and os.path.exists(video_file):
                            try:
                                processed_segments = self.facial_recognition.process_conversation_with_video(
                                    video_file=video_file,
                                    diarized_segments=segments,
                                    output_dir=self.conversations_dir,
                                    knowledge_base={},  
                                    workflow_state=self._get_state_from_history()
                                )
                                
                                if processed_segments:
                                    
                                    for segment in processed_segments:
                                        for history_segment in self.conversation_history:
                                            if (history_segment.get("speaker") == segment.get("speaker") and 
                                                history_segment.get("text") == segment.get("text")):
                                                history_segment["person"] = segment.get("person", history_segment.get("person"))
                                    
                                    self.current_video_buffer = []
                                
                            except Exception as e:
                                print(f"Error processing video with facial recognition: {e}")
                                traceback.print_exc()
                            finally:
                                if os.path.exists(video_file):
                                    os.unlink(video_file)
            else:
                transcription = self.transcriber.transcribe(audio_file)
                
                if self._is_repetitive_transcription(transcription):
                    print(f"Filtering repetitive transcription: {transcription}")
                    os.unlink(audio_file)  
                    return None, False
                
                if not transcription or not transcription.strip():
                    os.unlink(audio_file) 
                    return None, False
                
                self.conversation_history.append({
                    "timestamp": time.time(),
                    "speaker": "UNKNOWN",
                    "person": "UNKNOWN",
                    "text": transcription
                })
            
            os.unlink(audio_file) 
            
            if TRIGGER_PHRASE.lower() in transcription.lower():
                self._activate_listening_mode()
            elif STOP_PHRASE.lower() in transcription.lower():
                self._deactivate_listening_mode()
            
            current_speaker = self.conversation_history[-1].get("speaker", "UNKNOWN") if self.conversation_history else "UNKNOWN"
            if "let me think" in transcription.lower() and current_speaker.upper() == "USER":
                state = self._get_state_from_history()
                last_non_user_speaker = None
                for item in reversed(self.conversation_history[:-1]): 
                    if item.get("speaker", "").upper() != "USER":
                        last_non_user_speaker = item.get("speaker", "SPEAKER_1")
                        break
                
                if last_non_user_speaker:
                    suggestion = self.rizz_agent.generate_immediate_suggestion(
                        state=state,
                        speaker=last_non_user_speaker
                    )
                    
                    if suggestion:
                        print(f"\n💡 IMMEDIATE SUGGESTION: {suggestion}\n")
            
            if transcription.strip():
                print(f"Transcribed: {transcription}")
                return transcription, pause_detected
            
            return None, False
        except Exception as e:
            print(f"Audio processing error: {e}")
            traceback.print_exc()
            return None, False
    
    def _check_for_trigger_phrase(self, audio_file):
        """
        Fast check for trigger phrase to immediately activate listening mode.
        """
        try:
            quick_transcription = self.transcriber.transcribe(
                audio_file, 
                detect_trigger_only=True 
            )
            
            if not quick_transcription:
                return False
                
            if TRIGGER_PHRASE.lower() in quick_transcription.lower():
                self._activate_listening_mode()
                return True
                
            return False
        except Exception as e:
            print(f"Error checking for trigger phrase: {e}")
            return False
            
    def _activate_listening_mode(self):
        """
        Activate listening mode and immediately prepare the rizz agent.
        """
        if not self.active_listening:
            self.active_listening = True
            self.text_output.output("Active listening enabled.", "SYSTEM")
            print("\n🎧 ACTIVE LISTENING MODE ENABLED 🎧")
            
            try:
                self.rizz_agent._check_knowledge_base()
            except Exception as e:
                print(f"Warning: Could not check knowledge base: {e}")
            
            state = self._get_state_from_history()
            self.rizz_agent.prepare_for_active_listening(state)
                
    def _deactivate_listening_mode(self):
        """
        Deactivate listening mode.
        """
        if self.active_listening:
            self.active_listening = False
            self.text_output.output("Active listening disabled.", "SYSTEM")
            print("\n🛑 ACTIVE LISTENING MODE DISABLED 🛑")
    
    def _get_state_from_history(self):
        """
        Convert conversation history to a state dictionary for the rizz agent.
        """
        state = {
            "conversation": "",
            "speaker_segments": self.speaker_segments,
            "topics": [],
            "knowledge_base": {},
            "personal_info": []
        }
        
        recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        
        for item in recent_history:
            speaker = item.get("speaker", "UNKNOWN")
            
            if speaker.upper() == "USER":
                speaker = "USER"
            
            text = item.get("text", "").strip()
            if text:  # Only add non-empty messages
                state["conversation"] += f"[{speaker}]: {text}\n"
        
        return state
    
    def capture_user_reference(self):
        """
        Capture a reference sample of the user's voice for speaker diarization.
        """
        if not self.use_diarization:
            print("Speaker diarization is not enabled.")
            return False
            
        try:
            print("\n===== SPEAKER REFERENCE NEEDED =====")
            print("To differentiate between speakers, we need a sample of your voice.")
            print("Please speak continuously for 10 seconds when prompted.")
            
            time.sleep(1)  
            
            reference_path = self.recorder.capture_user_voice_reference(duration=10.0)
            
            if reference_path and os.path.exists(reference_path):
                if self.transcriber.set_user_reference(reference_path):
                    self.user_reference_captured = True
                    print("✓ User voice reference captured and processed successfully!")
                    return True
                else:
                    print("Failed to process user voice reference.")
                    return False
            else:
                print("Failed to capture user voice reference.")
                return False
                
        except Exception as e:
            print(f"Error capturing user reference: {e}")
            return False
    
    def _save_conversation_history(self):
        """
        Save the conversation history to a file.
        """
        try:
            output_dir = os.path.join(self.conversations_dir, "cursor_conversations")
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            file_path = os.path.join(output_dir, f"conversation-{timestamp}.txt")
            
            with open(file_path, "w") as f:
                f.write(f"Conversation recorded on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                
                for item in self.conversation_history:
                    speaker = item.get("speaker", "UNKNOWN")
                    person = item.get("person", speaker)
                    text = item.get("text", "")
                    timestamp = datetime.fromtimestamp(item.get("timestamp", time.time())).strftime('%H:%M:%S')
                    
                    f.write(f"[{timestamp}] [{person}]: {text}\n")
            
            print(f"Saved conversation history to {file_path}")
            return file_path
        except Exception as e:
            print(f"Error saving conversation history: {e}")
            traceback.print_exc()
            return None
    
    def active_listen(self):
        """
        In active listening, we continuously record and process audio.
        """
        print(f"[Active Listening Mode] Say '{STOP_PHRASE}' to exit active mode.")
        print(f"Response suggestions will appear every {self.rizz_agent.suggestion_cooldown} seconds.")
        print(f"Say 'let me think' at any time to get an immediate suggestion.")
        
        state = self._get_state_from_history()
        self.rizz_agent.prepare_for_active_listening(state)
        
        while self.active_listening and self.is_running:
            transcription, pause_detected = self.process_audio_segment(active_mode=True)
            
            time.sleep(0.1)
    
    def run(self):
        """
        The main loop. Continuously records audio in the background thread
        and processes it. When trigger phrase is detected, switches to active listening mode.
        """
        output_file = get_output_file()
        with open(output_file, "w") as f:
            f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        
        if self.use_diarization:
            self.capture_user_reference()
            
        if self.use_camera and self.facial_recognition:
            
            self.video_thread = threading.Thread(target=self._video_thread_func)
            self.video_thread.daemon = True
            self.video_thread.start()
        
        print(f"Say '{TRIGGER_PHRASE}' to enable Active Listening mode")
        print(f"Say '{STOP_PHRASE}' to exit Active Listening mode\n")
        print(f"Say 'let me think' at any time to get an immediate response suggestion\n")
    
        self.recorder.start_recording()
        
        self.is_running = True
        
        try:
            while self.is_running:
                if self.active_listening:
                    self.active_listen()
                else:
                    transcription, _ = self.process_audio_segment(active_mode=False)
                    
                    time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        except Exception as e:
            print(f"Error in main loop: {e}")
            traceback.print_exc()
        finally:
            self._save_conversation_history()
            
            try:
                if hasattr(self.recorder, 'stop_recording'):
                    print("Stopping audio recording...")
                    self.recorder.stop_recording()
            except Exception as e:
                print(f"Error stopping recording: {e}")
            
            self.video_recording = False
            if self.video_thread and self.video_thread.is_alive():
                print("Waiting for video thread to stop...")
                self.video_thread.join(timeout=2.0)
            
            print("Cursor Dating Assistant stopped.")
    
    def _video_thread_func(self):
        """
        Simplified video thread that handles screen capture for face detection.
        """
        self.video_recording = True
        print("Video recording thread started")
        
        try:
            import pyautogui
            import numpy as np
            
            while self.video_recording:
                try:
                    screenshot = pyautogui.screenshot()
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    self.current_video_buffer.append((frame, time.time()))
                    
                    max_buffer_size = 30
                    if len(self.current_video_buffer) > max_buffer_size:
                        self.current_video_buffer = self.current_video_buffer[-max_buffer_size:]
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error in video capture: {e}")
                    time.sleep(1.0)
        except ImportError:
            print("Error: Could not import required libraries for screen capture")
        except Exception as e:
            print(f"Error in video thread: {e}")
            traceback.print_exc()
            
        print("Video thread stopped")
    
    def _save_temp_video(self):
        """
        Simplified method to save current video buffer to a temporary file.
        """
        if not self.current_video_buffer:
            return None
            
        try:
            temp_dir = os.path.join(os.path.dirname(self.conversations_dir), "temp_files")
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_file = os.path.join(temp_dir, f"temp_video_{int(time.time())}.mp4")
            
            frame, _ = self.current_video_buffer[0]
            height, width, _ = frame.shape
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_file, fourcc, 10, (width, height))
            
            frames_written = 0
            for frame, _ in self.current_video_buffer:
                if frame is not None:
                    out.write(frame)
                    frames_written += 1
            
            out.release()
            
            if frames_written > 0:
                print(f"Saved {frames_written} frames to temporary video")
                return temp_file
            else:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                return None
                
        except Exception as e:
            print(f"Error saving temp video: {e}")
            return None

    def transcribe(self, audio_file: str, detect_trigger_only=False) -> str:
        headers = {
            "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}"
        }
        
        files = {
            "file": open(audio_file, "rb")
        }
        
        data = {
            "model": GROQ_WHISPER_MODEL
        }
        
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
        transcription = result.get("text", "")
        
        cleaned_text = transcription.strip().lower()
        
        if (cleaned_text == "thank you" or 
            cleaned_text.startswith("thank you") or 
            cleaned_text.endswith("thank you") or
            cleaned_text == "thank you." or
            "thank you" in cleaned_text):
            print(f"Filtering out hallucinated 'thank you': '{transcription}'")
            return ""
            
        if cleaned_text == "you" or cleaned_text in ["you.", "you?", "you!"]:
            print(f"Filtering out single word 'you': '{transcription}'")
            return ""
            
        confidence = result.get("confidence", 1.0)
        if confidence < 0.7:  # Adjust threshold as needed
            print(f"Low confidence transcription ({confidence}): {transcription}")
            return ""
        
        return transcription

    def _is_repetitive_transcription(self, transcription):
        """Check if a transcription is repetitively occurring or too short to be meaningful."""
        if not transcription:
            return False
            
        noise_words = ["you", "my", "me", "i", "a", "the", "um", "uh", "ah", "oh", "eh"]
        
        hallucination_phrases = ["thank you", "thanks", "thank", "you're welcome"]
        
        if isinstance(transcription, str):
            cleaned_text = transcription.strip().lower()
            
            for phrase in hallucination_phrases:
                if phrase in cleaned_text:
                    print(f"Filtering out hallucinated phrase: '{transcription}'")
                    return True
            
            words = cleaned_text.split()
            if len(words) == 1:
                word = words[0].rstrip('.!?,:;')
                if word in noise_words:
                    print(f"Filtering out noise word: '{transcription}'")
                    return True
                if len(word) < 3:
                    print(f"Filtering out short word: '{transcription}'")
                    return True
            
            if cleaned_text == "you" or cleaned_text.startswith("you ") or cleaned_text.endswith(" you") or " you " in cleaned_text:
                if len(cleaned_text) < 8:  # Only filter if it's a short phrase containing "you"
                    print(f"Filtering out 'you' phrase: '{transcription}'")
                    return True
            
        count = sum(1 for t in self.recent_transcriptions if t.lower() == transcription.lower())
        
        self.recent_transcriptions.append(transcription)
        if len(self.recent_transcriptions) > self.max_recent_transcriptions:
            self.recent_transcriptions.pop(0) 
            
        return count >= self.repetition_threshold

    def process_conversation(self, audio_file: str, segments: List[Dict], num_speakers: int = 2) -> List[Dict]:
        combined_segments = []
        current_segment = None
        
        for segment in segments:
            if current_segment is None:
                current_segment = segment.copy()
            elif segment["start"] - current_segment["end"] < 0.3: 
                current_segment["end"] = segment["end"]
                current_segment["text"] = current_segment["text"] + " " + segment["text"]
            else:
                combined_segments.append(current_segment)
                current_segment = segment.copy()
        
        if current_segment:
            combined_segments.append(current_segment)
        
        segments = combined_segments
        

def parse_arguments():
    parser = argparse.ArgumentParser(description="Cursor Assistant with Speech Diarization")
    
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
    
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()
        
        if args.debug:
            import logging
            logging.basicConfig(level=logging.DEBUG)
            
            def excepthook(exc_type, exc_value, exc_traceback):
                print("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
            sys.excepthook = excepthook
        
        
        if args.diarization:
            print(f"Speaker diarization enabled (expected speakers: {args.speakers})")
            print("You'll be asked to provide a 10-second voice sample for identification")
        else:
            print("Speaker diarization disabled")
            
        if args.screen:
            print("Screen capture enabled for facial recognition")
            print("Faces will be automatically detected and matched during conversation")
            print(f"Conversations will be saved in: {os.path.join(os.getcwd(), 'conversations')}/cursor_conversations/")
        
        assistant = CursorAssistant(
            use_diarization=args.diarization,
            expected_speakers=args.speakers,
            use_camera=args.screen
        )
            
        assistant.run()
        
    except KeyboardInterrupt:
        print("\nExiting program due to keyboard interrupt")
    except Exception as e:
        print(f"Startup error: {e}")
        if 'args' in locals() and args.debug:
            traceback.print_exc() 