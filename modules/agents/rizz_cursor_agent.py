#!/usr/bin/env python3

import os
import time
from typing import Dict, List, Any, Optional
from groq import Groq
from langchain_groq import ChatGroq
import datetime
import threading
from modules.config import GROQ_MODEL

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
        self.suggestion_cooldown = 20.0  # Increased to 20 seconds to prevent too frequent responses
        # Track the last few suggestions to avoid repetition
        self.recent_suggestions = []
        self.max_recent_suggestions = 5
        self.model = GROQ_MODEL  # Use the model from config.py
        
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
            return False
            
        current_time = time.time()
        time_since_last = current_time - self.last_suggestion_time
        
        # Check if we're still in cooldown period
        if time_since_last < self.suggestion_cooldown:
            remaining = self.suggestion_cooldown - time_since_last
            if remaining < 19.0:  # Only log if we're close to being ready again
                print(f"Waiting {remaining:.1f} more seconds before next suggestion...")
            return False

        # Check if the message is too short or repetitive
        if state:
            # Get the current message
            current_message = self._get_latest_message_from_speaker(state, speaker)
            
            # Check if message is too short
            if current_message and len(current_message.strip().split()) < 3:
                print(f"Message too short for suggestion: '{current_message}'")
                # Update last time but don't generate
                self.last_suggestion_time = current_time
                return False
                
            # Check for repetitive messages from the same speaker
            speaker_segments = state.get("speaker_segments", [])
            if len(speaker_segments) >= 2:
                # Get the last 3 messages from this speaker
                speaker_messages = []
                for segment in reversed(speaker_segments):
                    if segment.get("speaker") == speaker:
                        speaker_messages.append(segment.get("text", "").strip())
                    if len(speaker_messages) >= 3:
                        break
                
                # Check if the last 2+ messages are the same
                if len(speaker_messages) >= 2 and speaker_messages[0] == speaker_messages[1]:
                    print(f"Repetitive messages detected: '{speaker_messages[0]}'")
                    # Update last time but don't generate
                    self.last_suggestion_time = current_time
                    return False
            
        # If all conditions are met, update state and return True
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
            # Get the most recent message from the speaker
            most_recent_message = self._get_latest_message_from_speaker(state, speaker)
            
            # Get recent conversation for context
            recent_conversation = self._get_recent_conversation_context(state)
            print(f"RECENT CONVERSATION!!!!!!!!!!!!=====================")
            print(recent_conversation)
            
            # Dating-specific prompt for charismatic responses
            system_message = f"""
            You are an expert dating coach who helps people sound smooth, charismatic, and impressive on dates.
            
            The person's date just said: "{most_recent_message}"
            
            Recent conversation context:
            {recent_conversation}
            
            Generate ONE smooth, charismatic response that the person could say to sound impressive and create a connection.
            Your response should be:
            1. Natural and conversational (not scripted or cheesy)
            2. Show genuine interest and thoughtfulness
            3. Be slightly clever or witty but not over the top
            4. Be around 15-25 words maximum
            5. Something that would make their date think "wow, they're really cool/interesting"
            
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
                
            return f"💬 Smooth response: {suggestion}"
            
        except Exception as e:
            print(f"Error generating suggestion: {e}")
            import traceback
            traceback.print_exc()
            return f"💬 Try acknowledging what they said and asking a thoughtful follow-up question."
    
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