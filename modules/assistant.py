#!/usr/bin/env python3

import os
import time
import queue
import threading
import cv2

import numpy as np

from datetime import datetime
from modules.config import get_output_file, GROQ_WHISPER_MODEL
from modules.audio import AudioRecorder
from modules.speech_to_text import SpeechToText
from modules.output import TextOutput
from modules.workflow import ConversationWorkflow
from modules.facial_recognition import FacialRecognitionModule
from modules.agents import RizzCursorAgent

class ConversationAssistant:
    """
    Orchestrates the entire pipeline using continuous audio recording:
      - Background thread for audio recording
      - Processing thread for STT and agent execution
      - Speaker diarization for multi-speaker conversations
      - Facial recognition for identifying speakers (when camera is enabled)
    """
    def __init__(self, use_diarization=True, expected_speakers=2, use_camera=False):
        """
        Initialize the Conversation Assistant
        """
        self.transcriber = SpeechToText(model_name=GROQ_WHISPER_MODEL, use_diarization=use_diarization)
        self.workflow = ConversationWorkflow()
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
        print(f"Conversations will be saved in: {self.conversations_dir}")
        
        self.has_processed_audio = False
        
        if self.use_camera:
            try:
                self.facial_recognition = FacialRecognitionModule()
                print("Facial recognition module initialized")
            except Exception as e:
                print(f"Error initializing facial recognition: {e}. Proceeding without it.")
                self.facial_recognition = None
                self.use_camera = False
        
        self.processing_queue = queue.Queue()
        self.processing_thread = None
        self.processing_active = False
        
        # State
        self.is_running = False
        self.active_listening = False 

    def process_audio_segment(self, active_mode=False):
        """
        Grabs an audio snippet, transcribes it, updates the workflow.
        Returns (transcription, pause_detected).
        """
        try:
            audio_file, pause_detected = self.recorder.get_audio_segment(
                wait_for_pause=active_mode
            )
            
            if not audio_file:
                return None, False
            
            self.has_processed_audio = True
            
            # Transcribe
            if self.use_diarization and self.user_reference_captured:
                transcription, segments = self.transcriber.transcribe_with_speakers(
                    audio_file, 
                    num_speakers=self.expected_speakers
                )
                
                if not transcription or not transcription.strip():
                    os.unlink(audio_file)  # clean up temp
                    return None, False
                
                print(f"Transcribed with {len(segments)} speaker segments: {transcription[:100]}...")
                
                if self.workflow:
                    try:
                        self.workflow.update_conversation(transcription)
                        self.workflow.update_speaker_segments(segments)
                        print("✓ Successfully processed through workflow")
                    except Exception as e:
                        print(f"Error updating workflow: {e}")
                        import traceback
                        traceback.print_exc()
                
                if self.use_camera and self.facial_recognition and segments:
                    video_file = None
                    if self.current_video_buffer:
                        video_file = self._save_temp_video()
                        
                        if video_file and os.path.exists(video_file):
                            try:
                                workflow_state = None
                                if self.workflow and hasattr(self.workflow, 'state'):
                                    workflow_state = self.workflow.state
                                    
                                    knowledge_base = {}
                                    if workflow_state:
                                        knowledge_base = workflow_state.get('knowledge_base', {})
                                        print(f"Including knowledge base with {len(knowledge_base)} topics in conversation data")
                                        
                                        if knowledge_base:
                                            print("Knowledge base topics:")
                                            for topic, snippets in knowledge_base.items():
                                                print(f"  - {topic}: {len(snippets)} snippets")
                                
                                processed_segments = self.facial_recognition.process_conversation_with_video(
                                    video_file=video_file,
                                    diarized_segments=segments,
                                    output_dir=self.conversations_dir,
                                    knowledge_base=knowledge_base,
                                    workflow_state=workflow_state
                                )
                                
                                if processed_segments:
                                    print(f"Successfully processed {len(processed_segments)} segments with facial recognition")
                                    
                                    self.workflow.update_speaker_segments(processed_segments)
                                    
                                    self.current_video_buffer = []
                                    print("Cleared video buffer after face detection")
                                
                            except Exception as e:
                                print(f"Error processing video with facial recognition: {e}")
                                import traceback
                                traceback.print_exc()
                            finally:
                                if os.path.exists(video_file):
                                    os.unlink(video_file)
            else:
                transcription = self.transcriber.transcribe(audio_file)
                
                if not transcription or not transcription.strip():
                    os.unlink(audio_file)  
                    return None, False
                
                if self.workflow:
                    try:
                        self.workflow.update_conversation(transcription)
                        print("✓ Successfully processed through workflow")
                    except Exception as e:
                        print(f"Error updating workflow: {e}")
            
            os.unlink(audio_file) 
            
            if transcription.strip():
                print(f"Transcribed: {transcription}")
                
                return transcription, pause_detected
            
            return None, False
        except Exception as e:
            print(f"Audio processing error: {e}")
            import traceback
            traceback.print_exc()
            return None, False
    
    def capture_user_reference(self):
        """
        Capture a reference sample of the user's voice for speaker diarization.
        This should be called before starting the main recording loop.
        """
        if not self.use_diarization:
            return False
            
        try:
            
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
    
    def capture_face_reference(self):
        """
        Capture a reference sample of the user's face for facial recognition from the screen.
        This should be called before starting the main recording loop.
        """
        if not self.use_camera or not self.facial_recognition:
            print("Facial recognition is not enabled.")
            return False
            
        try:

            time.sleep(5) 
            
            try:
                import pyautogui
                import numpy as np
                screenshot = pyautogui.screenshot()
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except ImportError:
                print("Error: Could not import required libraries for screen capture")
                return False
            except Exception as e:
                print(f"Error capturing screen: {e}")
                return False
                
            try:
                display_frame = cv2.resize(frame, (800, 600))
                cv2.imshow("Face Reference - Press any key when ready", display_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Error displaying frame: {e}")
                
            if self.facial_recognition.add_face("USER", frame):
                self.face_reference_captured = True
                print("✓ User face reference captured from screen and processed successfully!")
                return True
            else:
                print("No face detected in the screen capture.")
                print("Please try again with your face clearly visible on screen.")
                return False
                
        except Exception as e:
            print(f"Error capturing face reference: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _video_thread_func(self):
        """Background thread for capturing screen frames"""
        self.video_recording = True
        
        try:
            try:
                import pyautogui
                import numpy as np
            except ImportError:
                import subprocess
                subprocess.call(["pip", "install", "pyautogui", "numpy", "pillow"])
                import pyautogui
                import numpy as np
                from PIL import Image
                print("Successfully installed screen capture dependencies")
            
            audio_detected = False
            face_detected = False
            current_face_name = None
            
            face_consistency_buffer = []
            required_consistent_detections = 3 
            
            consecutive_errors = 0
            max_consecutive_errors = 5
            
            while self.video_recording:
                try:
                    
                    if self.has_processed_audio and not audio_detected:
                        print("Audio detected, beginning to look for faces...")
                        audio_detected = True
                    
                    if audio_detected:
                        if not face_detected:
                            frame = self._capture_screen_frame()
                            if frame is None:
                                
                                if self.facial_recognition:
                                    try:
                                        face_results = self.facial_recognition.process_video_frame(frame)
                                        if face_results:
                                            face_info = face_results[0]  
                                            face_name = face_info["name"]
                                            face_embedding = face_info["embedding"]
                                            
                                            face_consistency_buffer.append(face_name)
                                            
                                            if len(face_consistency_buffer) >= required_consistent_detections:
                                                from collections import Counter
                                                face_counts = Counter(face_consistency_buffer)
                                                most_common_face, count = face_counts.most_common(1)[0]
                                                
                                                if count >= required_consistent_detections * 0.7:  
                                                    current_face_name = most_common_face
                                                    self.facial_recognition.update_current_face(most_common_face, face_embedding)
                                                    face_detected = True
                                                    
                                                    for segment in self.workflow.speaker_segments:
                                                        segment["person"] = current_face_name
                                                    
                                                    if hasattr(self.workflow, 'state'):
                                                        if "speaker_segments" in self.workflow.state:
                                                            for segment in self.workflow.state["speaker_segments"]:
                                                                segment["person"] = current_face_name
                                                else:
                                                    print(f"Face detections not consistent enough. Continuing to collect samples...")
                                            else:
                                                print(f"Detected face: {face_name} ({len(face_consistency_buffer)}/{required_consistent_detections} samples)")
                                    except Exception as e:
                                        print(f"Error in face detection: {e}")
                        
                        elif self.facial_recognition and self.facial_recognition.should_recheck_face():
                            print(f"Rechecking if current face is still {current_face_name}...")
                            
                            verification_samples = []
                            verification_attempts = 0
                            max_verification_attempts = 3
                            
                            while verification_attempts < max_verification_attempts:
                                frame = self._capture_screen_frame()
                                if frame is None:
                                    print(f"Skipping verification attempt {verification_attempts+1} - could not capture valid frame")
                                    verification_attempts += 1
                                    time.sleep(0.5)
                                    continue
                                    
                                try:
                                    face_results = self.facial_recognition.process_video_frame(frame)
                                    if face_results:
                                        face_info = face_results[0]
                                        face_name = face_info["name"] 
                                        face_embedding = face_info["embedding"]
                                        
                                        is_same = self.facial_recognition.is_same_face(face_embedding)
                                        verification_samples.append(is_same)
                                    else:
                                        verification_samples.append(False)
                                except Exception as e:
                                    print(f"Error during verification attempt {verification_attempts+1}: {e}")
                                    
                                verification_attempts += 1
                                time.sleep(0.5) 
                            
                            same_face_count = sum(verification_samples)
                            if same_face_count >= len(verification_samples) / 2:
                                self.facial_recognition.last_face_check_time = time.time()
                            else:
                                face_detected = False
                                face_consistency_buffer = []
                    
                    sleep_time = 0.1 if (audio_detected and not face_detected) else 0.5
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    consecutive_errors += 1
                    print(f"Error in video thread (error {consecutive_errors}/{max_consecutive_errors}): {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print("Too many consecutive errors in video thread, resetting...")
                        consecutive_errors = 0
                        self.current_video_buffer = []
                        time.sleep(1.0) 
                
        except Exception as e:
            print(f"Fatal error in video thread: {e}")
            import traceback
            traceback.print_exc()
            
        print("Video thread stopped")
    
    def _capture_screen_frame(self, target_width=None, target_height=None):
        """
        Helper method to capture a screen frame with robust error handling.
        Uses maximum screen resolution when target dimensions are None.
        """
        try:
            try:
                import pyautogui
                screen_width, screen_height = pyautogui.size()
                screenshot = pyautogui.screenshot()
            except Exception as e:
                print(f"PyAutoGUI screenshot error: {e}. Trying alternative method...")
                try:
                    import PIL.ImageGrab
                    screenshot = PIL.ImageGrab.grab()
                    screen_width, screen_height = screenshot.size
                except Exception as e2:
                    print(f"PIL ImageGrab error: {e2}")
                    return None
            
            if screenshot is None:
                return None
            
            frame = np.array(screenshot)
            
            if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                return None
            
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if target_width is not None and target_height is not None:
                frame = cv2.resize(frame, (target_width, target_height))
            else:
                print(f"Using maximum screen resolution: {screen_width}x{screen_height}")
            
            timestamp = time.time()
            self.current_video_buffer.append((frame, timestamp))
            
            max_buffer_size = 30  
            if len(self.current_video_buffer) > max_buffer_size:
                self.current_video_buffer = self.current_video_buffer[-max_buffer_size:]
                
            return frame
            
        except Exception as e:
            print(f"Error capturing screen frame: {e}")
            return None
    
    def _save_temp_video(self):
        """Save the current video buffer to a temporary file"""
        if not self.current_video_buffer:
            return None
            
        try:
            temp_dir = os.path.join(os.path.dirname(self.conversations_dir), "temp_files")
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_file = os.path.join(temp_dir, f"temp_video_{int(time.time())}.mp4")
            
            if len(self.current_video_buffer) == 0:
                print("Error: Video buffer is empty")
                return None
                
            frame, _ = self.current_video_buffer[0]
            if frame is None or frame.size == 0:
                return None
                
            height, width, _ = frame.shape
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_file, fourcc, 10, (width, height))
            
            if not out.isOpened():
                return None
            
            frames_written = 0
            for frame, _ in self.current_video_buffer:
                if frame is not None and frame.size > 0:
                    out.write(frame)
                    frames_written += 1
                
            out.release()
            
            if frames_written == 0:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                return None
                
            return temp_file
        except Exception as e:
            print(f"Error saving temporary video: {e}")
            import traceback
            traceback.print_exc()
            if 'temp_file' in locals() and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
            return None
    
    def _processing_thread_func(self):
        """Background thread for processing transcriptions without blocking audio recording"""
        self.processing_active = True
        
        consecutive_errors = 0
        
        print("Processing thread started")
        
        while self.is_running and self.processing_active:
            try:
                try:
                    transcription, pause_detected = self.processing_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                if not transcription or not transcription.strip():
                    continue
                    
                try:
                    processor_agent = self.workflow.processor_agent
                    process_result = processor_agent.process(transcription)
                    category = process_result.get("category", "SKIP") 
                    explanation = process_result.get("explanation", "")
                    
                    if category == "SKIP":
                        print(f"Skipping workflow processing for segment: {transcription[:50]}...")
                        
                        if self.workflow and hasattr(self.workflow, 'state'):
                            try:
                                if hasattr(self.workflow, 'graph') and self.workflow.graph:
                                    self.workflow.state["category"] = "SKIP"
                                    self.workflow.state["last_processed"] = transcription
                                    
                                    try:
                                        result = self.workflow.graph.invoke(self.workflow.state)
                                        if result:
                                            self.workflow.state = result
                                    except Exception as e:
                                        if "conversation" in self.workflow.state:
                                            if self.workflow.state["conversation"]:
                                                self.workflow.state["conversation"] += "\n" + transcription
                                            else:
                                                self.workflow.state["conversation"] = transcription
                            except Exception as e:
                                print(f"Error updating workflow state with skipped segment: {e}")
                        
                        continue 
                except Exception as e:
                    print(f"Error pre-checking input with processor: {e}")
                
                if self.workflow:
                    try:
                        self.workflow.state["category"] = category
                        self.workflow.update_conversation(transcription)
                        print("✓ Successfully processed through workflow")
                        consecutive_errors = 0  
                    except Exception as e:
                        consecutive_errors += 1
                        print(f"Error in workflow: {e}")
                        import traceback
                        traceback.print_exc()
                
            except Exception as e:
                print(f"Error in processing thread: {e}")
                import traceback
                traceback.print_exc()
                
        print("Processing thread ended")
                
    def run(self):
        """
        The main loop. Continuously records audio in the background thread
        and processes it.
        """
        self._setup_directories()
        
        output_file = get_output_file()
        with open(output_file, "w") as f:
            f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print("\nConversation Assistant ready.")
        print(f"Saving system logs to: {output_file}")
        
        if self.use_diarization:
            self.capture_user_reference()
            
        if self.use_camera and self.facial_recognition:
            
            os.makedirs(self.conversations_dir, exist_ok=True)
            
            self.video_thread = threading.Thread(target=self._video_thread_func)
            self.video_thread.daemon = True
            self.video_thread.start()
        
        self.recorder.start_recording()
        
        self.processing_thread = threading.Thread(target=self._processing_thread_func)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.is_running = True
        
        try:
            while self.is_running:
                transcription, pause_detected = self.process_audio_segment(active_mode=False)
                
                if transcription and transcription.strip():
                    self.processing_queue.put((transcription, pause_detected))
                    
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
            print("Conversation Assistant stopped.")
            
    def _setup_directories(self):
        """
        Ensures all required directories are properly set up at startup
        """
        os.makedirs(self.conversations_dir, exist_ok=True)
        
        system_logs_dir = os.path.join(self.conversations_dir, "system_logs")
        system_data_dir = os.path.join(self.conversations_dir, "system_data")
        faces_dir = os.path.join(self.conversations_dir, "faces")
        temp_dir = os.path.join(os.getcwd(), "temp_files")
        
        os.makedirs(system_logs_dir, exist_ok=True)
        os.makedirs(system_data_dir, exist_ok=True)
        os.makedirs(faces_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

    def stop(self):
        """
        Stop all background threads and clean up resources.
        """
        print("Stopping conversation assistant...")
        
        self.is_running = False
        self.processing_active = False
        if self.processing_thread and self.processing_thread.is_alive():
            print("Waiting for processing thread to stop...")
            self.processing_thread.join(timeout=2.0)
        
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
            
        if self.video_capture:
            try:
                self.video_capture.release()
            except:
                pass
        
        self._cleanup_directories()
        
        print("All resources cleaned up.")
    
    def _cleanup_directories(self):
        """
        Ensures system directories exist and moves any misplaced files to system folders.
        """
        try:
            system_logs_dir = os.path.join(self.conversations_dir, "system_logs")
            system_data_dir = os.path.join(self.conversations_dir, "system_data")
            faces_dir = os.path.join(self.conversations_dir, "faces")
            temp_dir = os.path.join(os.getcwd(), "temp_files")
            
            os.makedirs(system_logs_dir, exist_ok=True)
            os.makedirs(system_data_dir, exist_ok=True)
            os.makedirs(faces_dir, exist_ok=True)
            os.makedirs(temp_dir, exist_ok=True)
            
            for item in os.listdir(self.conversations_dir):
                item_path = os.path.join(self.conversations_dir, item)
                
                if os.path.isdir(item_path) or item in ["system_logs", "system_data", "faces", "temp_files"]:
                    continue
                
                if item.endswith(".txt"):
                    destination = os.path.join(system_logs_dir, item)
                    print(f"Moving {item} to system logs directory")
                    os.rename(item_path, destination)
                else:
                    destination = os.path.join(system_data_dir, item)
                    print(f"Moving file {item} to system data directory")
                    os.rename(item_path, destination)
                    
            print("Directory cleanup completed")
        except Exception as e:
            print(f"Error during directory cleanup: {e}")
            import traceback
            traceback.print_exc()
            