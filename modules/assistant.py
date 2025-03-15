#!/usr/bin/env python3

import os
import time
import queue
import threading
import cv2

import numpy as np

from datetime import datetime
from modules.config import TRIGGER_PHRASE, STOP_PHRASE, get_output_file
from modules.audio import AudioRecorder
from modules.speech_to_text import SpeechToText
from modules.output import TextOutput
from modules.workflow import ConversationWorkflow
from modules.facial_recognition import FacialRecognitionModule

class ConversationAssistant:
    """
    Orchestrates the entire pipeline using continuous audio recording:
      - Background thread for audio recording
      - Processing thread for STT and agent execution
      - Active Listening mode toggling
      - Speaker diarization for multi-speaker conversations
      - Facial recognition for identifying speakers (when camera is enabled)
    """
    def __init__(self, use_diarization=True, expected_speakers=2, use_camera=False):
        """
        Initialize the Conversation Assistant
        
        Args:
            use_diarization: Whether to use speaker diarization
            expected_speakers: Number of expected speakers when using diarization
            use_camera: Whether to capture screen for face detection
        """
        # Initialize basic components
        self.transcriber = SpeechToText(use_diarization=use_diarization)
        self.workflow = ConversationWorkflow()
        self.text_output = TextOutput()
        
        # Initialize audio recording
        self.recorder = AudioRecorder()
        
        # Setup for speaker diarization
        self.use_diarization = use_diarization
        self.expected_speakers = expected_speakers
        self.user_reference_captured = False
        
        # Setup for visual/facial recognition
        self.use_camera = use_camera
        self.facial_recognition = None
        self.video_capture = None
        self.video_thread = None
        self.video_recording = False
        self.current_video_buffer = []
        
        # Use current project directory for saving conversations
        self.conversations_dir = os.path.join(os.getcwd(), "conversations")
        os.makedirs(self.conversations_dir, exist_ok=True)
        print(f"Conversations will be saved in: {self.conversations_dir}")
        
        # Track if we've processed any audio - used for face detection triggering
        self.has_processed_audio = False
        
        # If camera/screen capture is enabled, initialize facial recognition
        if self.use_camera:
            try:
                self.facial_recognition = FacialRecognitionModule()
                print("Facial recognition module initialized")
            except Exception as e:
                print(f"Error initializing facial recognition: {e}. Proceeding without it.")
                self.facial_recognition = None
                self.use_camera = False
        
        # For background processing
        self.processing_queue = queue.Queue()
        self.processing_thread = None
        self.processing_active = False
        
        # State
        self.is_running = False
        self.active_listening = False

    def process_audio_segment(self, active_mode=False):
        """
        Grabs an audio snippet, transcribes it, updates the workflow.
        If the trigger phrase or stop phrase is found, toggles active listening.
        Returns (transcription, pause_detected).
        """
        try:
            # Get audio segment without interrupting recording
            audio_file, pause_detected = self.recorder.get_audio_segment(
                wait_for_pause=active_mode
            )
            
            if not audio_file:
                return None, False
            
            # Set flag that we've processed audio (for triggering face capture)
            self.has_processed_audio = True
            
            # Transcribe
            if self.use_diarization and self.user_reference_captured:
                transcription, segments = self.transcriber.transcribe_with_speakers(
                    audio_file, 
                    num_speakers=self.expected_speakers
                )
                
                # Ensure we have valid transcription
                if not transcription or not transcription.strip():
                    os.unlink(audio_file)  # clean up temp
                    return None, False
                
                print(f"Transcribed with {len(segments)} speaker segments: {transcription[:100]}...")
                
                # Process through the workflow immediately to ensure the agents receive the input
                if self.workflow:
                    try:
                        self.workflow.update_conversation(transcription)
                        self.workflow.update_speaker_segments(segments)
                        print("✓ Successfully processed through workflow")
                    except Exception as e:
                        print(f"Error updating workflow: {e}")
                        import traceback
                        traceback.print_exc()
                
                # If camera is enabled and we have facial recognition and segments:
                if self.use_camera and self.facial_recognition and segments:
                    # Create a temporary video file from current buffer
                    video_file = None
                    if self.current_video_buffer:
                        video_file = self._save_temp_video()
                        
                        if video_file and os.path.exists(video_file):
                            try:
                                # Get workflow state for additional context
                                workflow_state = None
                                if self.workflow and hasattr(self.workflow, 'state'):
                                    workflow_state = self.workflow.state
                                    
                                    # Get knowledge base from workflow state
                                    knowledge_base = {}
                                    if workflow_state:
                                        knowledge_base = workflow_state.get('knowledge_base', {})
                                        print(f"Including knowledge base with {len(knowledge_base)} topics in conversation data")
                                        
                                        # Log the knowledge base topics for debugging
                                        if knowledge_base:
                                            print("Knowledge base topics:")
                                            for topic, snippets in knowledge_base.items():
                                                print(f"  - {topic}: {len(snippets)} snippets")
                                
                                # Process video with diarized segments
                                processed_segments = self.facial_recognition.process_conversation_with_video(
                                    video_file=video_file,
                                    diarized_segments=segments,
                                    output_dir=self.conversations_dir,
                                    knowledge_base=knowledge_base,
                                    workflow_state=workflow_state
                                )
                                
                                # Check if we have processed segments
                                if processed_segments:
                                    print(f"Successfully processed {len(processed_segments)} segments with facial recognition")
                                    
                                    # Update the workflow with person information
                                    self.workflow.update_speaker_segments(processed_segments)
                                    
                                    # After processing, clear video buffer as we only need one face per conversation
                                    self.current_video_buffer = []
                                    print("Cleared video buffer after face detection")
                                
                            except Exception as e:
                                print(f"Error processing video with facial recognition: {e}")
                                import traceback
                                traceback.print_exc()
                            finally:
                                # Clean up temporary video file
                                if os.path.exists(video_file):
                                    os.unlink(video_file)
            else:
                transcription = self.transcriber.transcribe(audio_file)
                
                # Ensure we have valid transcription
                if not transcription or not transcription.strip():
                    os.unlink(audio_file)  # clean up temp
                    return None, False
                
                # Process through the workflow immediately 
                if self.workflow:
                    try:
                        self.workflow.update_conversation(transcription)
                        print("✓ Successfully processed through workflow")
                    except Exception as e:
                        print(f"Error updating workflow: {e}")
            
            os.unlink(audio_file)  # clean up temp
            
            if transcription.strip():
                print(f"Transcribed: {transcription}")
                
                # Check for trigger or stop phrases inline
                if TRIGGER_PHRASE.lower() in transcription.lower():
                    self.active_listening = True
                    self.text_output.output("Active listening enabled.", "SYSTEM")
                elif STOP_PHRASE.lower() in transcription.lower():
                    self.active_listening = False
                    self.text_output.output("Active listening disabled.", "SYSTEM")
                
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
            print("Speaker diarization is not enabled.")
            return False
            
        try:
            print("\n===== SPEAKER REFERENCE NEEDED =====")
            print("To differentiate between speakers, we need a sample of your voice.")
            print("Please speak continuously for 10 seconds when prompted.")
            
            time.sleep(1)  # Give user time to read
            
            # Capture user reference
            reference_path = self.recorder.capture_user_voice_reference(duration=10.0)
            
            if reference_path and os.path.exists(reference_path):
                # Set the reference in the transcriber
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
            print("\n===== FACE REFERENCE NEEDED =====")
            print("To identify you on screen, we need to capture your face from the screen.")
            print("Please make sure your face is clearly visible on screen (e.g., in a video call window).")
            print("Starting in 5 seconds...")
            
            time.sleep(5)  # Give user time to prepare
            
            # Capture screen
            try:
                import pyautogui
                import numpy as np
                screenshot = pyautogui.screenshot()
                frame = np.array(screenshot)
                # Convert RGB to BGR (OpenCV format)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except ImportError:
                print("Error: Could not import required libraries for screen capture")
                return False
            except Exception as e:
                print(f"Error capturing screen: {e}")
                return False
                
            # Show the captured frame
            try:
                # Resize for display
                display_frame = cv2.resize(frame, (800, 600))
                cv2.imshow("Face Reference - Press any key when ready", display_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Error displaying frame: {e}")
                # Continue anyway since we have the frame
                
            # Try to add the face to the database
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
            # Check if we have the required libraries
            try:
                import pyautogui
                import numpy as np
                from PIL import Image
            except ImportError:
                print("Error: pyautogui library not found. Installing required packages...")
                import subprocess
                subprocess.call(["pip", "install", "pyautogui", "numpy", "pillow"])
                import pyautogui
                import numpy as np
                from PIL import Image
                print("Successfully installed screen capture dependencies")
            
            # Get screen size
            screen_width, screen_height = pyautogui.size()
            target_width, target_height = 640, 480
            
            print(f"Screen capture initialized with resolution {screen_width}x{screen_height}")
            print(f"Downsampling to {target_width}x{target_height} for processing")
            print("Waiting for audio to start capturing the screen...")
            
            # Set up variables for face tracking
            audio_detected = False
            face_detected = False
            current_face_name = None
            last_face_check_time = time.time()
            
            # Record frames
            consecutive_errors = 0
            max_consecutive_errors = 5
            
            while self.video_recording:
                try:
                    current_time = time.time()
                    
                    # STEP 1: Wait for audio before we start trying to detect faces
                    if self.has_processed_audio and not audio_detected:
                        print("Audio detected, beginning to look for faces...")
                        audio_detected = True
                    
                    # Only proceed if we've detected audio
                    if audio_detected:
                        # STEP 2: If we haven't found a face yet, try to find one
                        if not face_detected:
                            print("Looking for a face to track...")
                            # Capture screen with robust error handling
                            frame = self._capture_screen_frame(target_width, target_height)
                            if frame is None:
                                raise Exception("Failed to capture valid screen frame")
                                
                            # Try to detect faces in the current frame
                            if self.facial_recognition:
                                try:
                                    face_results = self.facial_recognition.process_video_frame(frame)
                                    if face_results:
                                        face_info = face_results[0]  # Get first (largest) detected face
                                        face_name = face_info["name"]
                                        face_embedding = face_info["embedding"]
                                        
                                        # Store the current face information
                                        current_face_name = face_name
                                        self.facial_recognition.update_current_face(face_name, face_embedding)
                                        face_detected = True
                                        
                                        print(f"✓ Face detected! Identified as: {face_name}")
                                        print(f"The same person will be used for the entire conversation")
                                        print(f"Face will be rechecked every {self.facial_recognition.get_recheck_interval()} seconds")
                                        
                                        # For all future audio segments, associate with this person
                                        for segment in self.workflow.speaker_segments:
                                            segment["person"] = current_face_name
                                        
                                        # Update the workflow state
                                        if hasattr(self.workflow, 'state'):
                                            if "speaker_segments" in self.workflow.state:
                                                for segment in self.workflow.state["speaker_segments"]:
                                                    segment["person"] = current_face_name
                                except Exception as e:
                                    print(f"Error in face detection: {e}")
                        
                        # STEP 3: If we already have a face, only check periodically if it's still the same person
                        elif self.facial_recognition and self.facial_recognition.should_recheck_face():
                            print(f"Rechecking if current face is still {current_face_name}...")
                            # Capture a new frame
                            frame = self._capture_screen_frame(target_width, target_height)
                            if frame is None:
                                print("Skipping face recheck - could not capture valid frame")
                            else:
                                # Process the frame to see if we still have the same face
                                try:
                                    face_results = self.facial_recognition.process_video_frame(frame)
                                    if face_results:
                                        face_info = face_results[0]
                                        face_name = face_info["name"] 
                                        face_embedding = face_info["embedding"]
                                        
                                        # Check if it's the same face
                                        is_same = self.facial_recognition.is_same_face(face_embedding)
                                        
                                        if is_same:
                                            print(f"✓ Verified: Still the same person ({current_face_name})")
                                            # Update the last check time
                                            self.facial_recognition.last_face_check_time = time.time()
                                        else:
                                            # We have a different person now
                                            print(f"⚠ Different person detected! Updating from {current_face_name} to {face_name}")
                                            current_face_name = face_name
                                            self.facial_recognition.update_current_face(face_name, face_embedding)
                                            
                                            # For all future segments, use the new person
                                            # (we don't change past segments to maintain conversation history)
                                    else:
                                        print("No face found during recheck - keeping current face identification")
                                except Exception as e:
                                    print(f"Error during face recheck: {e}")
                    
                    # Sleep to reduce CPU usage - shorter if we're actively looking for a face
                    sleep_time = 0.1 if (audio_detected and not face_detected) else 0.5
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    consecutive_errors += 1
                    print(f"Error in video thread (error {consecutive_errors}/{max_consecutive_errors}): {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print("Too many consecutive errors in video thread, resetting...")
                        consecutive_errors = 0
                        self.current_video_buffer = []
                        time.sleep(1.0)  # Sleep longer after errors
                
        except Exception as e:
            print(f"Fatal error in video thread: {e}")
            import traceback
            traceback.print_exc()
            
        print("Video thread stopped")
    
    def _capture_screen_frame(self, target_width=640, target_height=480):
        """Helper method to capture a screen frame with robust error handling"""
        try:
            # Try primary method using pyautogui
            try:
                import pyautogui
                screenshot = pyautogui.screenshot()
            except Exception as e:
                print(f"PyAutoGUI screenshot error: {e}. Trying alternative method...")
                # Alternative method using PIL directly
                try:
                    import PIL.ImageGrab
                    screenshot = PIL.ImageGrab.grab()
                except Exception as e2:
                    print(f"PIL ImageGrab error: {e2}")
                    return None
            
            # Ensure screenshot is a valid image
            if screenshot is None:
                return None
            
            # Convert to numpy array (OpenCV format)
            frame = np.array(screenshot)
            
            # Check if frame is valid
            if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                return None
            
            # Convert RGB to BGR (OpenCV format)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Resize frame to reduce memory usage
            frame = cv2.resize(frame, (target_width, target_height))
            
            # Store frame in buffer (keep just enough frames for face detection)
            timestamp = time.time()
            self.current_video_buffer.append((frame, timestamp))
            
            # Limit buffer size to avoid memory issues
            max_buffer_size = 30  # Just need a few frames to find a face
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
            # Create temp file in a proper system temp directory
            # Create a temp directory first to ensure it's properly managed
            temp_dir = os.path.join(os.path.dirname(self.conversations_dir), "temp_files")
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_file = os.path.join(temp_dir, f"temp_video_{int(time.time())}.mp4")
            
            # Get frame properties
            if len(self.current_video_buffer) == 0:
                print("Error: Video buffer is empty")
                return None
                
            frame, _ = self.current_video_buffer[0]
            if frame is None or frame.size == 0:
                print("Error: Invalid frame in buffer")
                return None
                
            height, width, _ = frame.shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_file, fourcc, 10, (width, height))
            
            if not out.isOpened():
                print(f"Error: Could not open video writer for {temp_file}")
                return None
            
            # Write frames
            frames_written = 0
            for frame, _ in self.current_video_buffer:
                if frame is not None and frame.size > 0:
                    out.write(frame)
                    frames_written += 1
                
            # Release writer
            out.release()
            
            if frames_written == 0:
                print("Error: No frames were written to the video file")
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                return None
                
            print(f"Successfully saved {frames_written} frames to temporary video")
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
        max_consecutive_errors = 3
        
        print("Processing thread started")
        
        while self.is_running and self.processing_active:
            try:
                # Get an item with timeout to allow checking is_running
                try:
                    transcription, pause_detected = self.processing_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Skip empty transcriptions
                if not transcription or not transcription.strip():
                    continue
                    
                # Print what we're processing
                print(f"Processing queue item: {transcription[:50]}..." if len(transcription) > 50 else transcription)
                
                # First, check if this is meaningful input using the processor agent
                try:
                    processor_agent = self.workflow.processor_agent
                    process_result = processor_agent.process(transcription)
                    category = process_result.get("category", "SKIP") 
                    explanation = process_result.get("explanation", "")
                    print(f"Processing as: {category} - {explanation}")
                    
                    # If it's SKIP category, don't process through the full workflow
                    if category == "SKIP":
                        print("Skipping full workflow processing for meaningless input")
                        continue
                        
                except Exception as e:
                    print(f"Error pre-checking input with processor: {e}")
                    # Continue with normal processing as fallback
                
                # Process through the workflow
                if self.workflow:
                    try:
                        # Try running the workflow to process the input
                        self.workflow.update_conversation(transcription)
                        print("✓ Successfully processed through workflow")
                        consecutive_errors = 0  # Reset error counter
                    except Exception as e:
                        consecutive_errors += 1
                        print(f"Error in workflow: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        # if consecutive_errors >= max_consecutive_errors:
                        #     print(f"Too many consecutive errors ({consecutive_errors}), resetting workflow")
                        #     # Consider resetting the workflow or state here
                        #     self.workflow.state = CONVERSATION_STATE_SCHEMA.copy()
                        #     consecutive_errors = 0
                
            except Exception as e:
                print(f"Error in processing thread: {e}")
                import traceback
                traceback.print_exc()
                
        print("Processing thread ended")
                
    def active_listen(self):
        """
        In active listening, we continuously record and process audio.
        When a pause is detected, a response is generated.
        """
        print(f"[Active Listening Mode] Say '{STOP_PHRASE}' to exit active mode.")
        while self.active_listening and self.is_running:
            self.process_audio_segment(active_mode=True)
            time.sleep(0.1)

    def run(self):
        """
        The main loop. Continuously records audio in the background thread
        and processes it. If user says TRIGGER_PHRASE, we switch to 
        active listening until STOP_PHRASE.
        """
        # Ensure all directories are properly set up
        self._setup_directories()
        
        # Create a new conversation file for this session
        output_file = get_output_file()
        with open(output_file, "w") as f:
            f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print("\nConversation Assistant ready.")
        print(f"Saving system logs to: {output_file}")
        
        # If speaker diarization is enabled, capture user reference first
        if self.use_diarization:
            self.capture_user_reference()
            
        # If screen capture is enabled, start it without requiring face reference
        if self.use_camera and self.facial_recognition:
            print("\n===== FACIAL RECOGNITION ENABLED =====")
            print("Faces will be automatically detected and saved as conversations occur.")
            print("No face reference needed - faces will be automatically identified and matched.")
            
            # Create conversations directory
            os.makedirs(self.conversations_dir, exist_ok=True)
            
            # Start video recording thread
            self.video_thread = threading.Thread(target=self._video_thread_func)
            self.video_thread.daemon = True
            self.video_thread.start()
        
        print(f"Say '{TRIGGER_PHRASE}' to enable Active Listening mode")
        print(f"Say '{STOP_PHRASE}' to exit Active Listening mode\n")

        # Start the background recording thread
        self.recorder.start_recording()
        
        # Start the processing thread
        self.processing_thread = threading.Thread(target=self._processing_thread_func)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.is_running = True
        
        # Main loop
        try:
            while self.is_running:
                # Process audio segment
                transcription, pause_detected = self.process_audio_segment(
                    active_mode=self.active_listening
                )
                
                # If we have transcription, queue it for processing
                if transcription and transcription.strip():
                    self.processing_queue.put((transcription, pause_detected))
                    
                # Adjust sleep time based on active listening mode
                sleep_time = 0.1 if self.active_listening else 0.5
                time.sleep(sleep_time)
                
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
        # Set up main conversations directory
        os.makedirs(self.conversations_dir, exist_ok=True)
        
        # Set up system subdirectories
        system_logs_dir = os.path.join(self.conversations_dir, "system_logs")
        system_data_dir = os.path.join(self.conversations_dir, "system_data")
        temp_dir = os.path.join(os.path.dirname(self.conversations_dir), "temp_files")
        
        os.makedirs(system_logs_dir, exist_ok=True)
        os.makedirs(system_data_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        print(f"Directory structure initialized:")
        print(f"- Conversations: {self.conversations_dir}")
        print(f"- System logs: {system_logs_dir}")
        print(f"- System data: {system_data_dir}")
        print(f"- Temporary files: {temp_dir}")

    def stop(self):
        """
        Stop all background threads and clean up resources.
        """
        print("Stopping conversation assistant...")
        
        # Stop processing thread
        self.is_running = False
        self.processing_active = False
        if self.processing_thread and self.processing_thread.is_alive():
            print("Waiting for processing thread to stop...")
            self.processing_thread.join(timeout=2.0)
        
        # Stop recording
        try:
            if hasattr(self.recorder, 'stop_recording'):
                print("Stopping audio recording...")
                self.recorder.stop_recording()
        except Exception as e:
            print(f"Error stopping recording: {e}")
        
        # Stop video resources
        self.video_recording = False
        if self.video_thread and self.video_thread.is_alive():
            print("Waiting for video thread to stop...")
            self.video_thread.join(timeout=2.0)
            
        if self.video_capture:
            try:
                self.video_capture.release()
            except:
                pass
        
        # Save final conversation data for all recognized persons
        self._save_final_conversation_data()
        
        # Organize directories and clean up any misplaced files
        self._cleanup_directories()
        
        print("All resources cleaned up.")
    
    def _cleanup_directories(self):
        """
        Ensures all conversation data is properly organized in person folders.
        Moves any misplaced files to the appropriate person directory or system folders.
        """
        try:
            print("\nCleaning up and organizing conversation directories...")
            # Ensure system directories exist
            system_logs_dir = os.path.join(self.conversations_dir, "system_logs")
            system_data_dir = os.path.join(self.conversations_dir, "system_data")
            os.makedirs(system_logs_dir, exist_ok=True)
            os.makedirs(system_data_dir, exist_ok=True)
            
            # Get all files in the main conversations directory
            for item in os.listdir(self.conversations_dir):
                item_path = os.path.join(self.conversations_dir, item)
                
                # Skip directories and system folders
                if os.path.isdir(item_path) or item in ["system_logs", "system_data", "temp_files"]:
                    continue
                
                # Handle misplaced files
                if item.startswith("conversation_") or item.endswith(".txt"):
                    # Move conversation files to system logs
                    destination = os.path.join(system_logs_dir, item)
                    print(f"Moving {item} to system logs directory")
                    os.rename(item_path, destination)
                elif item.endswith(".pkl"):
                    # Move database files to system data
                    destination = os.path.join(system_data_dir, item)
                    print(f"Moving {item} to system data directory")
                    os.rename(item_path, destination)
                else:
                    # Unsure about file type, move to system data
                    destination = os.path.join(system_data_dir, item)
                    print(f"Moving unknown file {item} to system data directory")
                    os.rename(item_path, destination)
                    
            print("Directory cleanup completed")
        except Exception as e:
            print(f"Error during directory cleanup: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_final_conversation_data(self):
        """
        Save the final conversation data for all recognized persons.
        Uses a simplified structure with just one directory per person and three text files:
        1. conversation.txt - Chat history
        2. knowledge_base.txt - Knowledge base for interests
        3. topics.txt - Topics of interest
        
        Ensures consistent directory naming to avoid duplicate directories.
        """
        print("\nSaving final conversation data...")
        
        try:
            # Get the current state from workflow
            if not self.workflow or not hasattr(self.workflow, 'state'):
                print("No workflow state available to save")
                return
                
            state = self.workflow.state
            
            # Get knowledge base and conversation history
            knowledge_base = state.get('knowledge_base', {})
            conversation = state.get('conversation', '')
            topics = state.get('topics', [])
            speaker_segments = state.get('speaker_segments', [])
            
            # Get unique persons from the speaker segments
            unique_persons = set()
            for segment in speaker_segments:
                if 'person' in segment and segment['person']:
                    unique_persons.add(segment['person'])
            
            # If no persons identified, use "Unknown" as fallback with a unique timestamp
            if not unique_persons:
                # Check if we've already created a default person ID
                if hasattr(self, '_default_person_id'):
                    unique_persons = {self._default_person_id}
                else:
                    current_time = int(time.time())
                    self._default_person_id = f"Person_{current_time}"
                    unique_persons = {self._default_person_id}
                
            print(f"Saving conversation data for {len(unique_persons)} recognized persons: {', '.join(unique_persons)}")
                
            # Create conversations directory if it doesn't exist
            os.makedirs(self.conversations_dir, exist_ok=True)
                
            # Dictionary to keep track of clean person names
            person_id_map = {}
                
            # Save data for each unique person
            for person in unique_persons:
                # Ensure a consistent person name
                if person.startswith("Person_") or person.startswith("Unknown"):
                    # Extract just the base identifier without multiple timestamps
                    parts = person.split("_")
                    if len(parts) > 1:
                        # Keep just the first segment (Person or Unknown) and the next segment
                        base_name = f"{parts[0]}_{parts[1]}"
                        person_id_map[person] = base_name
                        person = base_name
                    elif "_" not in person[7:]:  # No timestamp yet
                        # Keep consistent with workflow by using the same ID
                        if hasattr(self.workflow, '_default_person_id'):
                            person = self.workflow._default_person_id
                        else:
                            # If all else fails, use the existing name
                            pass
                
                person_dir = os.path.join(self.conversations_dir, person)
                os.makedirs(person_dir, exist_ok=True)
                
                # 1. Save conversation history
                conversation_path = os.path.join(person_dir, "conversation.txt")
                
                # Check if file exists to append or create new
                file_mode = 'a' if os.path.exists(conversation_path) else 'w'
                
                with open(conversation_path, file_mode) as f:
                    # Add timestamp if appending
                    if file_mode == 'a':
                        f.write(f"\n\n--- CONTINUED CONVERSATION {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")
                    else:
                        f.write(f"CONVERSATION WITH {person.upper()}\n")
                        f.write("="*80 + "\n\n")
                    
                    # First add the structured speaker segments if available
                    if speaker_segments:
                        for segment in speaker_segments:
                            speaker = segment.get('speaker', 'Unknown')
                            person_name = segment.get('person', speaker)
                            # Use cleaned name if available
                            if person_name in person_id_map:
                                person_name = person_id_map[person_name]
                            text = segment.get('text', '')
                            f.write(f"[{person_name}]: {text}\n")
                    else:
                        # Otherwise add the raw conversation text
                        f.write(conversation)
                
                # 2. Save knowledge base if available
                if knowledge_base:
                    kb_path = os.path.join(person_dir, "knowledge_base.txt")
                    
                    # Check if file exists to append or create new
                    kb_file_mode = 'a' if os.path.exists(kb_path) else 'w'
                    
                    with open(kb_path, kb_file_mode) as f:
                        # Add timestamp if appending
                        if kb_file_mode == 'a':
                            f.write(f"\n\n--- UPDATED KNOWLEDGE BASE {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")
                        else:
                            f.write(f"KNOWLEDGE BASE FOR {person.upper()}\n")
                            f.write("="*80 + "\n\n")
                        
                        for topic, snippets in knowledge_base.items():
                            f.write(f"TOPIC: {topic}\n")
                            f.write("-"*80 + "\n")
                            for i, snippet in enumerate(snippets):
                                f.write(f"{i+1}. {snippet}\n\n")
                            f.write("\n")
                
                # 3. Save topics of interest if available
                if topics:
                    topics_path = os.path.join(person_dir, "topics.txt")
                    
                    # Check if file exists to append or create new
                    topics_file_mode = 'a' if os.path.exists(topics_path) else 'w'
                    
                    with open(topics_path, topics_file_mode) as f:
                        # Add timestamp if appending
                        if topics_file_mode == 'a':
                            f.write(f"\n\n--- UPDATED TOPICS {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")
                        else:
                            f.write(f"TOPICS OF INTEREST FOR {person.upper()}\n")
                            f.write("="*80 + "\n\n")
                        
                        for topic in topics:
                            if isinstance(topic, dict):
                                f.write(f"• {topic.get('name', '')}: {topic.get('description', '')}\n")
                            else:
                                f.write(f"• {topic}\n")
                
                print(f"Saved data for {person} in {person_dir}")
                
        except Exception as e:
            print(f"Error saving final conversation data: {e}")
            import traceback
            traceback.print_exc() 