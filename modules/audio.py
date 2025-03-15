#!/usr/bin/env python3

import os
import time
import queue
import tempfile
import wave
import threading
import numpy as np
import sounddevice as sd
from typing import Tuple, Optional, Dict, List

from modules.config import SAMPLE_RATE, CHANNELS, BUFFER_SIZE, SILENCE_THRESHOLD
from modules.config import CONVERSATION_PAUSE, CONTINUOUS_PROCESSING_INTERVAL

class AudioRecorder:
    """
    Records audio continuously in a dedicated thread.
    Captures audio data without interruption while allowing periodic processing.
    """
    def __init__(self, sample_rate=SAMPLE_RATE, channels=CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.audio_buffer = queue.Queue()
        self.processing_buffer = queue.Queue()
        self.silence_counter = 0
        self.pause_detected = False
        self.last_process_time = 0
        self.stream = None
        self.record_thread = None
        self.user_reference_path = None
        self.capturing_user_reference = False
        self.user_reference_duration = 10.0  # 10 seconds for user voice reference
        self.user_reference_data = []
        
    def callback(self, indata, frames, time_info, status):
        """Callback function for the sounddevice stream"""
        if status:
            print(f"Stream status: {status}")
        
        # Always put data in the main audio buffer
        self.audio_buffer.put(indata.copy())
        
        # Detect silence
        volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
        if volume_norm < SILENCE_THRESHOLD:
            self.silence_counter += frames / self.sample_rate
            if self.silence_counter >= CONVERSATION_PAUSE:
                self.pause_detected = True
        else:
            self.silence_counter = 0
            self.pause_detected = False
    
    def start_recording(self):
        """Start continuous recording in a separate thread"""
        if self.recording:
            return  # Already recording
        
        self.recording = True
        self.record_thread = threading.Thread(target=self._record_thread_func)
        self.record_thread.daemon = True 
        self.record_thread.start()
        print("Audio recording started")
    
    def stop_recording(self):
        """Stop the recording thread"""
        self.recording = False
        if self.record_thread:
            self.record_thread.join(timeout=1.0)
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("Recording stopped")
    
    def _record_thread_func(self):
        """Main function for the recording thread"""
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.callback,
                blocksize=BUFFER_SIZE
            )
            
            self.stream.start()
            while self.recording:
                time.sleep(0.1) 
            
            self.stream.stop()
            self.stream.close()
            self.stream = None
        except Exception as e:
            print(f"Recording thread error: {e}")
            self.recording = False
    
    def get_audio_segment(self, duration=CONTINUOUS_PROCESSING_INTERVAL, wait_for_pause=False) -> Tuple[str, bool]:
        """
        Get a segment of audio for processing.
        """
        if not self.recording:
            return "", False
        
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_filename = temp_file.name
        temp_file.close()
        
        collected_audio = []
        start_time = time.time()
        found_pause = False
        
        while self.recording:
            # Try to get data with a short timeout
            try:
                data = self.audio_buffer.get(timeout=0.1)
                collected_audio.append(data)
            except queue.Empty:
                pass
            
            # Check if we've collected enough audio
            elapsed = time.time() - start_time
            if wait_for_pause:
                if self.pause_detected and elapsed >= 1.0:  # At least 1 second of audio
                    found_pause = True
                    break
                elif elapsed >= duration * 2: 
                    break
            else:
                if elapsed >= duration:
                    break
        
        if collected_audio:
            try:
                audio_data = np.concatenate(collected_audio, axis=0)
                with wave.open(temp_filename, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self.sample_rate)
                    wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
                return temp_filename, found_pause
            except Exception as e:
                print(f"Error saving audio: {e}")
                os.unlink(temp_filename)
                return "", False
        else:
            os.unlink(temp_filename)
            return "", False 

    def capture_user_voice_reference(self, duration: float = 10.0) -> Optional[str]:
        """
        Capture a sample of the user's voice to use as a reference for speaker diarization.
        
        Args:
            duration: How long to record the user's voice (in seconds)
            
        Returns:
            Path to the saved user reference file, or None if failed
        """
        if self.recording:
            print("Already recording. Please stop recording first.")
            return None
            
        print(f"\n===== CAPTURING USER VOICE REFERENCE =====")
        print(f"Please speak continuously for {duration} seconds...")
        print(f"This will help the assistant identify your voice in conversations.")
        
        # Start recording specifically for user reference
        self.user_reference_duration = duration
        self.user_reference_data = []
        self.capturing_user_reference = True
        
        # Create a temporary file for the user reference
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.user_reference_path = temp_file.name
        temp_file.close()
        
        # Set up the recording stream
        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self._user_reference_callback,
                blocksize=BUFFER_SIZE
            )
            
            stream.start()
            start_time = time.time()
            
            # Display a countdown
            while time.time() - start_time < duration:
                remaining = duration - (time.time() - start_time)
                print(f"\rRecording user reference: {remaining:.1f} seconds remaining...", end="")
                time.sleep(0.1)
                
            print("\r\nFinished recording user reference!")
            stream.stop()
            stream.close()
            
            # Save the recording
            if self.user_reference_data:
                try:
                    audio_data = np.concatenate(self.user_reference_data, axis=0)
                    with wave.open(self.user_reference_path, 'wb') as wf:
                        wf.setnchannels(self.channels)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(self.sample_rate)
                        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
                    
                    print(f"User reference saved to: {self.user_reference_path}")
                    return self.user_reference_path
                except Exception as e:
                    print(f"Error saving user reference: {e}")
                    if os.path.exists(self.user_reference_path):
                        os.unlink(self.user_reference_path)
                    self.user_reference_path = None
                    return None
            else:
                print("No audio data captured for user reference")
                return None
                
        except Exception as e:
            print(f"Error capturing user reference: {e}")
            self.capturing_user_reference = False
            if os.path.exists(self.user_reference_path):
                os.unlink(self.user_reference_path)
            self.user_reference_path = None
            return None
        finally:
            self.capturing_user_reference = False
    
    def _user_reference_callback(self, indata, frames, time_info, status):
        """Callback for capturing user reference audio"""
        if status:
            print(f"User reference stream status: {status}")
        self.user_reference_data.append(indata.copy()) 