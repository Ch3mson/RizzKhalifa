#!/usr/bin/env python3

"""
Test script for speaker diarization.
This script will:
1. Record a 10-second reference sample from the user
2. Record a 20-second audio clip (which should contain multiple speakers)
3. Transcribe the audio with speaker diarization
"""

import os
import time
import tempfile
import argparse
import shutil
from modules.audio import AudioRecorder
from modules.speech_to_text import SpeechToText

def main():
    parser = argparse.ArgumentParser(description="Test Speaker Diarization")
    parser.add_argument("--speakers", type=int, default=2,
                      help="Expected number of speakers (default: 2)")
    parser.add_argument("--save-recordings", action="store_true",
                      help="Save the recordings for later analysis")
    parser.add_argument("--user-ref", type=str, default=None,
                      help="Path to existing user reference recording (skip recording step)")
    parser.add_argument("--conversation", type=str, default=None,
                      help="Path to existing conversation recording (skip recording step)")
    parser.add_argument("--threshold", type=float, default=0.6,
                      help="Similarity threshold for user identification (default: 0.6)")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug output")
    args = parser.parse_args()
    
    print("\n==== Speaker Diarization Test ====")
    
    if not args.user_ref and not args.conversation:
        print("This test will record audio in two stages:")
        print("1. A 10-second reference sample of your voice")
        print("2. A 20-second conversation with multiple speakers")
        print("Then it will transcribe the conversation with speaker labels.")
        
        if args.save_recordings:
            print("\nNOTE: Recordings will be saved to the current directory for future testing.")
    
    recorder = AudioRecorder()
    transcriber = SpeechToText(use_diarization=True)
    
    user_reference_path = args.user_ref
    if not user_reference_path:
        print("\n===== Step 1: Capturing User Reference =====")
        print("Please speak continuously for 10 seconds...")
        user_reference_path = recorder.capture_user_voice_reference(duration=10.0)
        
        if not user_reference_path:
            print("Failed to capture user reference. Exiting.")
            return
        
        if args.save_recordings:
            saved_ref_path = "user_reference.wav"
            shutil.copy(user_reference_path, saved_ref_path)
            print(f"Saved user reference to: {os.path.abspath(saved_ref_path)}")
    else:
        print(f"\n===== Using existing user reference: {user_reference_path} =====")
    
    if args.threshold != 0.6:
        print(f"Using custom similarity threshold: {args.threshold}")
        if hasattr(transcriber, 'diarization_agent') and transcriber.diarization_agent:
            transcriber.diarization_agent.similarity_threshold = args.threshold
    
    if not transcriber.set_user_reference(user_reference_path):
        print("Failed to process user reference. Exiting.")
        return
    
    conversation_path = args.conversation
    if not conversation_path:
        print("\n===== Step 2: Recording Conversation =====")
        print("Now record a conversation with multiple speakers for 20 seconds...")
        print("Ready to record in 3 seconds...")
        time.sleep(3)
        
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        conversation_path = temp_file.name
        temp_file.close()
        
        conversation_data = []
        
        def conversation_callback(indata, frames, time_info, status):
            if status:
                print(f"Conversation recording status: {status}")
            conversation_data.append(indata.copy())
        
        import sounddevice as sd
        import numpy as np
        import wave
        from modules.config import SAMPLE_RATE, CHANNELS, BUFFER_SIZE
        
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=conversation_callback,
            blocksize=BUFFER_SIZE
        )
        
        stream.start()
        start_time = time.time()
        duration = 20.0  # 20 seconds for conversation
        
        while time.time() - start_time < duration:
            remaining = duration - (time.time() - start_time)
            print(f"\rRecording conversation: {remaining:.1f} seconds remaining...", end="")
            time.sleep(0.1)
        
        print("\r\nFinished recording conversation!")
        stream.stop()
        stream.close()
        
        if conversation_data:
            try:
                audio_data = np.concatenate(conversation_data, axis=0)
                with wave.open(conversation_path, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
                
                print(f"Conversation saved to: {conversation_path}")
                
                if args.save_recordings:
                    saved_conv_path = "conversation.wav"
                    shutil.copy(conversation_path, saved_conv_path)
                    print(f"Saved conversation to: {os.path.abspath(saved_conv_path)}")
            except Exception as e:
                print(f"Error saving conversation: {e}")
                return
        else:
            print("No audio data captured for conversation")
            return
    else:
        print(f"\n===== Using existing conversation: {conversation_path} =====")
    
    # Enable debug mode
    if args.debug and hasattr(transcriber, 'diarization_agent') and transcriber.diarization_agent:
        # Enable more detailed debugging in the diarization agent
        print("\n===== Debug mode enabled =====")
    
    print("\n===== Step 3: Transcribing with Speaker Diarization =====")
    transcript, segments = transcriber.transcribe_with_speakers(
        conversation_path, 
        num_speakers=args.speakers
    )
    
    print("\n===== Transcription Result =====")
    print(transcript)
    
    print("\n===== Detailed Segments =====")
    for i, segment in enumerate(segments):
        print(f"{i+1}. [{segment['speaker']}] {segment['start']:.2f}s - {segment['end']:.2f}s: {segment['text']}")
    
    if not args.save_recordings:
        print("\nCleaning up temporary files...")
        try:
            if not args.user_ref:
                os.unlink(user_reference_path)
            if not args.conversation:
                os.unlink(conversation_path)
        except Exception as e:
            print(f"Error cleaning up: {e}")
    else:
        print("\nTemporary files not deleted since --save-recordings was specified")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 