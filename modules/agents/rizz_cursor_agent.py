#!/usr/bin/env python3

import os
import time
from typing import Dict, List, Any, Optional
from groq import Groq
from langchain_groq import ChatGroq
import threading
import requests
from supabase import create_client
from dotenv import load_dotenv
from modules.config import GROQ_MODEL

# Load environment variables from .env file
load_dotenv()

class RizzCursorAgent:
    """
    Agent that helps the user sound smooth and charismatic in conversations,
    especially in dating contexts. Provides subtle, impressive responses that
    make the user appear interesting and thoughtful.
    
    This agent is only activated when:
    1. The user is in active listening mode
    2. A non-user speaker has just spoken
    
    It provides witty, charismatic responses that the user can use or adapt
    to impress the other person in the conversation.
    """
    
    def __init__(self):
        """Initialize the RizzCursorAgent with Groq client."""
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.last_speaker = None
        self.last_suggestion_time = 0
        self.suggestion_cooldown = 10.0  # Changed from 20 seconds to 10 seconds as requested
        # Track the last few suggestions to avoid repetition
        self.recent_suggestions = []
        self.max_recent_suggestions = 5
        self.model = GROQ_MODEL  # Use the model from config.py
        self.first_response_generated = False
        self.active_listening_started = False
        self.mp3_generation_time = 0  # Track when an MP3 was last generated
        
        # For voice generation
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.use_voice = True  # Flag to enable/disable voice generation
        self.voice_model = "tts-1"  # OpenAI's text-to-speech model
        self.voice_name = "alloy"  # Natural sounding voice
        self.voice_speed = 1.3  # Faster voice speech (default is 1.0)
        
        # Supabase client setup for storage
        supabase_url = os.environ.get("SUPABASE_URL", "")
        supabase_key = os.environ.get("SUPABASE_KEY", "")
        self.use_supabase = supabase_url and supabase_key
        
        if self.use_supabase:
            try:
                self.supabase_client = create_client(supabase_url, supabase_key)
                print("Supabase client initialized for voice file uploads")
            except Exception as e:
                print(f"Error initializing Supabase client: {e}")
                self.use_supabase = False
        
        # Make sure cursor_messages directory exists
        os.makedirs("cursor_messages", exist_ok=True)
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def prepare_for_active_listening(self, state: Dict[str, Any]):
        """
        When active listening mode is activated, prepare for contextual responses.
        
        Args:
            state: The current conversation state
        """
        print("Rizz agent ready for smooth conversational responses...")
        print(f"Will suggest responses with a {self.suggestion_cooldown} second cooldown period")
        
        # Mark that active listening has started and generate an immediate response
        self.active_listening_started = True
        
        # Generate initial response based on existing conversation
        self._generate_initial_response(state)
    
    def _generate_initial_response(self, state: Dict[str, Any]):
        """
        Generate an initial response as soon as active listening is enabled.
        
        Args:
            state: The current conversation state
        """
        try:
            # Get the entire conversation history so far
            entire_conversation = state.get("conversation", "")
            
            if not entire_conversation:
                return
                
            print("Generating initial response for active listening mode...")
            
            # Dating-specific prompt for initial response
            system_message = f"""
            You are an expert dating coach who helps people sound smooth, charismatic, and impressive on dates.
            
            Here's the conversation so far:
            {entire_conversation}
            
            Based on what has been said so far, generate ONE smooth, charismatic response that the person could say
            to sound impressive and create a connection. Consider ALL the conversation so far, not just the last message.
            
            Your response should be:
            1. Natural and conversational (not scripted or cheesy)
            2. Show genuine interest and thoughtfulness
            3. Be slightly clever or witty but not over the top
            4. Be around 30 words maximum
            5. Something that would make their date think "wow, they're really cool/interesting"
            6. ALWAYS respond in English, regardless of input language
            
            DO NOT use greeting formats like "You could say:" or "Try this:". Just provide the exact response to use.
            """
            
            # Generate the initial suggestion
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Give me one smooth, charismatic response based on this conversation."}
                ],
                temperature=0.8,
                max_tokens=90,
                top_p=1.0
            )
            
            suggestion = response.choices[0].message.content.strip()
            
            # Clean the suggestion
            if suggestion.startswith('"') and suggestion.endswith('"'):
                suggestion = suggestion[1:-1]
                
            # Remove any "You could say:" type prefixes
            prefixes_to_remove = [
                "You could say:", "You could try:", "Try saying:", "Say something like:", 
                "Here's a response:", "Response:", "You might respond:"
            ]
            for prefix in prefixes_to_remove:
                if suggestion.startswith(prefix):
                    suggestion = suggestion[len(prefix):].strip()
            
            # Save to recent suggestions and mark first response as generated
            self.recent_suggestions.append(suggestion)
            self.first_response_generated = True
            self.last_suggestion_time = time.time()
            
            # Generate voice file
            voice_file = self._generate_voice_file(suggestion)
            
            print(f"💬 Initial smooth response: {suggestion}")
            if voice_file:
                print(f"🔊 Voice file generated: {voice_file}")
            
        except Exception as e:
            print(f"Error generating initial suggestion: {e}")
            import traceback
            traceback.print_exc()
            
    def is_ready_to_generate(self, speaker: str, active_listening: bool = False, state: Dict[str, Any] = None) -> bool:
        """
        Check if we should generate a suggestion based on current speaker and mode.
        
        Args:
            speaker: The speaker ID of the current speaker
            active_listening: Whether active listening mode is enabled
            state: Optional state to check message content
            
        Returns:
            bool: True if we should generate a suggestion
        """
        # Only generate if:
        # 1. Active listening is enabled
        # 2. Current speaker is not the user (so we're helping user respond to someone else)
        # 3. We haven't just generated a suggestion (cooldown)
        
        if not active_listening:
            return False
            
        is_user = speaker.upper() == "USER"
        if is_user:
            # Check if user said "let me think" - if so, generate immediately
            if state:
                current_message = self._get_latest_message_from_speaker(state, speaker)
                if current_message and "let me think" in current_message.lower():
                    print("Detected 'let me think' trigger phrase. Generating immediate response...")
                    return True
            return False
            
        current_time = time.time()
        time_since_last = current_time - self.last_suggestion_time
        
        # Check if we need to wait after MP3 generation
        time_since_mp3 = current_time - self.mp3_generation_time
        if self.mp3_generation_time > 0 and time_since_mp3 < 15.0:
            remaining = 15.0 - time_since_mp3
            print(f"Waiting {remaining:.1f} more seconds after MP3 generation...")
            return False
        
        # Skip cooldown check if this is the first response
        if self.first_response_generated:
            # Check if we're still in cooldown period
            if time_since_last < self.suggestion_cooldown:
                remaining = self.suggestion_cooldown - time_since_last
                if remaining < 9.0:  # Only log if we're close to being ready again (adjusted for 10s cooldown)
                    print(f"Waiting {remaining:.1f} more seconds before next suggestion...")
                return False
        
        # Always update state and return True to generate a response, even for short messages
        print(f"Ready to generate response after {time_since_last:.1f} seconds")
        self.last_speaker = speaker
        self.last_suggestion_time = current_time
        return True
        
    def generate_suggestion(self, 
                           state: Dict[str, Any],
                           speaker: str) -> Optional[str]:
        """
        Generate a smooth, charismatic response for the user to say on their date.
        
        Args:
            state: The conversation state containing conversation history
            speaker: The current speaker ID (who the user needs to respond to)
            
        Returns:
            str: A witty, thoughtful response that makes the user sound impressive
        """
        try:
            # Get the entire conversation history
            entire_conversation = state.get("conversation", "")
            
            # Get the most recent message from the speaker to directly respond to
            most_recent_message = self._get_latest_message_from_speaker(state, speaker)
            
            # Check if the most recent message is just filler or too short
            is_filler_message = self._is_filler_message(most_recent_message)
            
            # Get context from multiple recent messages for context
            recent_context = self._get_combined_recent_messages(state, 3)  # Get up to 3 recent messages for context
            
            # Dating-specific prompt for charismatic responses
            system_message = f"""
            You are an expert dating coach who helps people sound smooth, charismatic, and impressive on dates.
            
            Full conversation history:
            {entire_conversation}
            
            The person's date just said: "{most_recent_message}"
            
            Recent conversation context:
            {recent_context}
            
            Generate ONE smooth, charismatic response that the person could say to sound impressive and create a connection.
            
            {"IMPORTANT: The most recent message is just filler or very short. Respond to the overall conversation context instead." if is_filler_message else "IMPORTANT: Respond directly to what they just said in a relevant way."}
            
            Your response should be:
            1. Natural and conversational (not scripted or cheesy)
            2. Show genuine interest and thoughtfulness
            3. Be slightly clever or witty but not over the top
            4. Be around 15-25 words maximum
            5. Something that would make their date think "wow, they're really cool/interesting"
            6. ALWAYS respond in English, regardless of input language
            
            DO NOT use greeting formats like "You could say:" or "Try this:". Just provide the exact response to use.
            """
            
            # Generate the suggestion using the configured model
            response = self.client.chat.completions.create(
                model=self.model,  # Use the model from config
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Give me one smooth, charismatic response to what they just said."}
                ],
                temperature=0.8,  # Slightly higher for more creative responses
                max_tokens=60,
                top_p=1.0
            )
            
            suggestion = response.choices[0].message.content.strip()
            
            # Clean the suggestion
            if suggestion.startswith('"') and suggestion.endswith('"'):
                suggestion = suggestion[1:-1]
                
            # Remove any "You could say:" type prefixes
            prefixes_to_remove = [
                "You could say:", "You could try:", "Try saying:", "Say something like:", 
                "Here's a response:", "Response:", "You might respond:"
            ]
            for prefix in prefixes_to_remove:
                if suggestion.startswith(prefix):
                    suggestion = suggestion[len(prefix):].strip()
            
            # Save to recent suggestions to avoid repetition
            self.recent_suggestions.append(suggestion)
            if len(self.recent_suggestions) > self.max_recent_suggestions:
                self.recent_suggestions.pop(0)
            
            # Mark that we've generated the first response
            if not self.first_response_generated:
                self.first_response_generated = True
            
            # Generate voice file
            voice_file = self._generate_voice_file(suggestion)
            
            result_text = f"💬 Smooth response: {suggestion}"
            if voice_file:
                result_text += f"\n🔊 Voice file generated: {voice_file}"
                
            return result_text
            
        except Exception as e:
            print(f"Error generating suggestion: {e}")
            import traceback
            traceback.print_exc()
            return f"💬 Try acknowledging what they said and asking a thoughtful follow-up question."
    
    def _generate_voice_file(self, text: str) -> Optional[str]:
        """
        Generate an MP3 voice file for the given text.
        
        Args:
            text: The text to convert to speech
            
        Returns:
            str: Path to the generated MP3 file or None if generation failed
        """
        if not self.use_voice or not text or not self.openai_api_key:
            return None
            
        try:
            timestamp = int(time.time())
            filename = f"cursor_messages/message_{timestamp}.mp3"

            # print("======== FILE NAME =========")
            # print(filename)
            # print("============================")
            supabase_path = f"audio/message_{timestamp}.mp3"
                    
            # First try using Groq's API through their client
            try:
                # Check if Groq supports TTS via system capabilities
                prompt = f"""
                Convert the following text to speech. Return a base64 encoded MP3 file:
                
                "{text}"
                """
                
                # Try to use Groq for TTS, but will likely fall back to OpenAI 
                # as Groq doesn't have a dedicated TTS endpoint yet
                groq_tts_possible = False
                
                if not groq_tts_possible:
                    # Fall back to OpenAI's TTS API
                    headers = {
                        "Authorization": f"Bearer {self.openai_api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    data = {
                        "model": self.voice_model,
                        "input": text,
                        "voice": self.voice_name,
                        "speed": self.voice_speed  # Add speed parameter for faster speech
                    }
                    
                    response = requests.post(
                        "https://api.openai.com/v1/audio/speech",
                        headers=headers,
                        json=data
                    )
                    
                    if response.status_code == 200:
                        # Save the audio file locally
                        with open(filename, "wb") as f:
                            f.write(response.content)
                        
                        # Upload to Supabase if enabled
                        if self.use_supabase:
                            try:
                                # Upload the MP3 file to Supabase storage
                                with open(filename, "rb") as f:
                                    upload_response = (
                                        self.supabase_client.storage
                                        .from_("llm")  # Bucket name for audio files
                                        .upload(
                                            file=f,
                                            path=supabase_path,
                                            file_options={"cache-control": "3600", "upsert": "true"}
                                        )
                                    )
                                print(f"Uploaded voice file to Supabase: {supabase_path}")
                            except Exception as e:
                                print(f"Error uploading voice file to Supabase: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        # Update MP3 generation time to enforce waiting period
                        self.mp3_generation_time = time.time()
                        print(f"MP3 generated at {self.mp3_generation_time}, will wait 10 seconds before next response")
                        
                        return filename
                    else:
                        print(f"Error generating voice with OpenAI: {response.text}")
                        return None
            except Exception as e:
                print(f"Error in voice generation: {e}")
                return None
                
        except Exception as e:
            print(f"Error generating voice file: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _is_filler_message(self, message: str) -> bool:
        """
        Check if a message is just filler content like "um" or "uh" or too short to respond to.
        
        Args:
            message: The message to check
            
        Returns:
            bool: True if the message is just filler
        """
        if not message:
            return True
            
        # Strip punctuation and whitespace
        clean_message = message.strip().lower()
        
        # Check if it's too short (fewer than 2 words)
        if len(clean_message.split()) < 2:
            return True
            
        # Check if it's just filler words
        filler_patterns = [
            "um", "uh", "er", "hmm", "ah", "oh", "like", "you know", 
            "actually", "basically", "so", "well", "just"
        ]
        
        # Check if the message consists primarily of filler words
        word_count = 0
        filler_count = 0
        
        for word in clean_message.split():
            word_count += 1
            if word in filler_patterns:
                filler_count += 1
                
        # If more than 50% of the words are fillers and less than 4 total words, it's a filler message
        return (filler_count / max(word_count, 1)) > 0.5 and word_count < 4
    
    def _get_combined_recent_messages(self, state: Dict[str, Any], num_messages: int = 3) -> str:
        """Get multiple recent messages combined into a single context string."""
        try:
            speaker_segments = state.get("speaker_segments", [])
            if not speaker_segments:
                return state.get("conversation", "")
                
            # Get last N segments, regardless of speaker
            recent_segments = speaker_segments[-min(len(speaker_segments), num_messages):]
            
            # Process in chronological order (oldest to newest)
            conversation = ""
            for seg in recent_segments:
                speaker = seg.get("speaker", "Unknown")
                text = seg.get("text", "").strip()
                if text:  # Only include non-empty messages
                    conversation += f"[{speaker}]: {text}\n"
                
            return conversation
        except Exception as e:
            print(f"Error getting combined recent messages: {e}")
            return ""
    
    def _get_recent_conversation_context(self, state: Dict[str, Any]) -> str:
        """Get recent conversation segments for better context."""
        try:
            speaker_segments = state.get("speaker_segments", [])
            if not speaker_segments:
                return state.get("conversation", "")
                
            # Get last 5 segments
            recent_segments = speaker_segments[-5:]
            
            # Format them
            conversation = ""
            for seg in recent_segments:
                speaker = seg.get("speaker", "Unknown")
                text = seg.get("text", "")
                conversation += f"[{speaker}]: {text}\n"
                
            return conversation
        except Exception as e:
            print(f"Error getting conversation context: {e}")
            return ""
    
    def _analyze_conversation_topics(self, state: Dict[str, Any]) -> tuple:
        """
        Analyze the conversation to extract relevant topics and knowledge.
        No longer used for generation but kept for compatibility.
        """
        try:
            # We're no longer using topics and knowledge base for generation
            # Returning empty strings for compatibility
            return "", ""
            
        except Exception as e:
            print(f"Error analyzing conversation topics: {e}")
            return "", ""
    
    def _get_latest_message_from_speaker(self, state: Dict[str, Any], speaker: str) -> str:
        """Get the most recent message from a specific speaker."""
        try:
            speaker_segments = state.get("speaker_segments", [])
            
            # Filter segments by speaker and get the most recent one
            for segment in reversed(speaker_segments):
                if segment.get("speaker") == speaker:
                    return segment.get("text", "")
                    
            return ""
        except Exception as e:
            print(f"Error getting latest message: {e}")
            return ""
    
    def generate_immediate_suggestion(self, state: Dict[str, Any], speaker: str) -> Optional[str]:
        """
        Generate a response immediately, bypassing all cooldown restrictions.
        Used specifically for the "let me think" trigger command.
        
        Args:
            state: The conversation state containing conversation history
            speaker: The current speaker ID (who the user needs to respond to)
            
        Returns:
            str: A witty, thoughtful response that makes the user sound impressive
        """
        try:
            print("Generating immediate response, bypassing cooldown...")
            # Temporarily store old cooldown and reset after
            old_cooldown = self.last_suggestion_time
            
            # Force suggestion by resetting cooldown
            self.last_suggestion_time = 0
            
            # Generate suggestion
            suggestion = self.generate_suggestion(state, speaker)
            
            # Restore previous cooldown
            self.last_suggestion_time = old_cooldown
            
            return suggestion
        except Exception as e:
            print(f"Error generating immediate suggestion: {e}")
            import traceback
            traceback.print_exc()
            return f"💬 Try acknowledging what they said and asking a thoughtful follow-up question."
                    
    def _get_recent_conversation(self, conversation: str, max_tokens: int = 1000) -> str:
        """Extract the most recent part of the conversation."""
        if not conversation:
            return ""
            
        # Simple approach: get last N characters
        return conversation[-min(len(conversation), max_tokens*4):]
        
    def _get_latest_segments(self, segments: List[Dict[str, Any]], num_segments: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent speaker segments."""
        if not segments:
            return []
            
        # Return the last N segments
        return segments[-min(len(segments), num_segments):] 