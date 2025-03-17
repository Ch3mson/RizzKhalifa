#!/usr/bin/env python3

import os
import time
from typing import Dict, List, Any, Optional
from groq import Groq
import threading
import requests
from supabase import create_client
from dotenv import load_dotenv
from modules.config import GROQ_MODEL

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
        self.suggestion_cooldown = 15.0  
        self.recent_suggestions = []
        self.max_recent_suggestions = 5
        self.model = GROQ_MODEL  
        self.first_response_generated = False
        self.active_listening_started = False
        self.mp3_generation_time = 0  
        
        # For voice generation
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.use_voice = True  
        self.voice_model = "tts-1"  
        self.voice_name = "alloy" 
        self.voice_speed = 1.3  
        
        # For knowledge base
        self.knowledge_base_size_threshold = 25000  
        self.knowledge_base_content = None 
        
        try:
            self._check_knowledge_base()
        except Exception as e:
            print(f"Warning: Failed to check knowledge base: {e}")
        
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
        
        os.makedirs("cursor_messages", exist_ok=True)
        
        self.lock = threading.Lock()
    
    def prepare_for_active_listening(self, state: Dict[str, Any]):
        """
        When active listening mode is activated, prepare for contextual responses.
        """
        
        self.active_listening_started = True
        
        current_time = time.time()
        time_since_last = current_time - max(self.last_suggestion_time, self.mp3_generation_time)
        
        if time_since_last >= self.suggestion_cooldown:
            print("Generating initial response for active listening mode...")
            self._generate_initial_response(state)
        else:
            remaining = self.suggestion_cooldown - time_since_last
            print(f"Cooldown active. Waiting {remaining:.1f} more seconds before generating initial response...")
    
    def _generate_initial_response(self, state: Dict[str, Any]):
        """
        Generate an initial response as soon as active listening is enabled.
        """
        try:
            entire_conversation = state.get("conversation", "")
            
            if not entire_conversation:
                return
                
            # print("Generating initial response for active listening mode...")
            
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
            
            if suggestion.startswith('"') and suggestion.endswith('"'):
                suggestion = suggestion[1:-1]
                
            prefixes_to_remove = [
                "You could say:", "You could try:", "Try saying:", "Say something like:", 
                "Here's a response:", "Response:", "You might respond:"
            ]
            for prefix in prefixes_to_remove:
                if suggestion.startswith(prefix):
                    suggestion = suggestion[len(prefix):].strip()
            
            self.recent_suggestions.append(suggestion)
            self.first_response_generated = True
            self.last_suggestion_time = time.time()
            
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
        """
        
        if not active_listening:
            return False
            
        is_user = speaker.upper() == "USER"
        if is_user:
            if state:
                current_message = self._get_latest_message_from_speaker(state, speaker)
                if current_message and "let me think" in current_message.lower():
                    print("Detected 'let me think' trigger phrase. Generating immediate response...")
                    return True
            return False
            
        current_time = time.time()
        
        last_activity_time = max(self.last_suggestion_time, self.mp3_generation_time)
        time_since_last = current_time - last_activity_time
        
        if time_since_last < self.suggestion_cooldown:
            remaining = self.suggestion_cooldown - time_since_last
            print(f"Waiting {remaining:.1f} more seconds before next suggestion... (last activity at {last_activity_time})")
            return False
        
        self.last_speaker = speaker
        self.last_suggestion_time = current_time
        return True
        
    def generate_suggestion(self, 
                           state: Dict[str, Any],
                           speaker: str) -> Optional[str]:
        """
        Generate a response for the user to say on their date.
        """
        try:
            self._check_knowledge_base()
            
            entire_conversation = state.get("conversation", "")
            
            most_recent_message = self._get_latest_message_from_speaker(state, speaker)
            
            is_filler_message = self._is_filler_message(most_recent_message)
            
            recent_context = self._get_combined_recent_messages(state, 3)  
            
            knowledge_base_context = ""
            if self.knowledge_base_content:
                knowledge_base_context = f"""
                ### User Knowledge Base:
                {self.knowledge_base_content}
                """
            
            system_message = f"""
            You are an expert dating coach who helps people sound smooth, charismatic, and impressive on dates.
            
            Full conversation history:
            {entire_conversation}
            
            The person's date just said: "{most_recent_message}"
            
            Recent conversation context:
            {recent_context}
            
            {knowledge_base_context}
            
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
            
            response = self.client.chat.completions.create(
                model=self.model, 
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Give me one smooth, charismatic response to what they just said."}
                ],
                temperature=0.8,  
                max_tokens=60,
                top_p=1.0
            )
            
            suggestion = response.choices[0].message.content.strip()
            
            if suggestion.startswith('"') and suggestion.endswith('"'):
                suggestion = suggestion[1:-1]
                
            prefixes_to_remove = [
                "You could say:", "You could try:", "Try saying:", "Say something like:", 
                "Here's a response:", "Response:", "You might respond:"
            ]
            for prefix in prefixes_to_remove:
                if suggestion.startswith(prefix):
                    suggestion = suggestion[len(prefix):].strip()
            
            self.recent_suggestions.append(suggestion)
            if len(self.recent_suggestions) > self.max_recent_suggestions:
                self.recent_suggestions.pop(0)
            
            if not self.first_response_generated:
                self.first_response_generated = True
            
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
        """
        if not self.use_voice or not text or not self.openai_api_key:
            return None
            
        try:
            timestamp = int(time.time())
            filename = f"cursor_messages/message_{timestamp}.mp3"
            supabase_path = f"audio/message_{timestamp}.mp3"
                    
            sentiment = self._analyze_sentiment(text)
                
            try:
                prompt = f"""
                Convert the following text to speech. Return a base64 encoded MP3 file:
                
                "{text}"
                """
                groq_tts_possible = False
                
                if not groq_tts_possible:
                    headers = {
                        "Authorization": f"Bearer {self.openai_api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    data = {
                        "model": self.voice_model,
                        "input": text,
                        "voice": self.voice_name,
                        "speed": self.voice_speed 
                    }
                    
                    response = requests.post(
                        "https://api.openai.com/v1/audio/speech",
                        headers=headers,
                        json=data
                    )
                    
                    if response.status_code == 200:
                        with open(filename, "wb") as f:
                            f.write(response.content)
                        
                        if self.use_supabase:
                            try:
                                with open(filename, "rb") as f:
                                    upload_response = (
                                        self.supabase_client.storage
                                        .from_("llm")  
                                        .upload(
                                            file=f,
                                            path=supabase_path,
                                            file_options={"cache-control": "3600", "upsert": "true"}
                                        )
                                    )
                                print(f"Uploaded voice file to Supabase: {supabase_path}")
                                
                                try:
                                    db_response = (
                                        self.supabase_client.table("agent-audio")
                                        .insert({
                                            "file_url": supabase_path,
                                            "text_content": text,
                                            "file_name": filename,
                                            "sentiment": sentiment
                                        })
                                        .execute()
                                    )
                                    print(f"Inserted record with sentiment analysis into agent-audio database table")
                                except Exception as e:
                                    print(f"Error inserting record into database: {e}")
                                    import traceback
                                    traceback.print_exc()
                                
                            except Exception as e:
                                print(f"Error uploading voice file to Supabase: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        self.mp3_generation_time = time.time()
                        print(f"MP3 generated at {self.mp3_generation_time}, will wait {self.suggestion_cooldown} seconds before next response")
                        
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
        """
        if not message:
            return True
            
        clean_message = message.strip().lower()
        
        if len(clean_message.split()) < 2:
            return True
            
        filler_patterns = [
            "um", "uh", "er", "hmm", "ah", "oh", "like", "you know", 
            "actually", "basically", "so", "well", "just"
        ]
        
        word_count = 0
        filler_count = 0
        
        for word in clean_message.split():
            word_count += 1
            if word in filler_patterns:
                filler_count += 1
                
        return (filler_count / max(word_count, 1)) > 0.5 and word_count < 4
    
    def _get_combined_recent_messages(self, state: Dict[str, Any], num_messages: int = 3) -> str:
        """Get multiple recent messages combined into a single context string."""
        try:
            speaker_segments = state.get("speaker_segments", [])
            if not speaker_segments:
                return state.get("conversation", "")
                
            recent_segments = speaker_segments[-min(len(speaker_segments), num_messages):]
            
            conversation = ""
            for seg in recent_segments:
                speaker = seg.get("speaker", "Unknown")
                text = seg.get("text", "").strip()
                if text: 
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
                
            recent_segments = speaker_segments[-5:]
            
            conversation = ""
            for seg in recent_segments:
                speaker = seg.get("speaker", "Unknown")
                text = seg.get("text", "")
                conversation += f"[{speaker}]: {text}\n"
                
            return conversation
        except Exception as e:
            print(f"Error getting conversation context: {e}")
            return ""
    
    def _get_latest_message_from_speaker(self, state: Dict[str, Any], speaker: str) -> str:
        """Get the most recent message from a specific speaker."""
        try:
            speaker_segments = state.get("speaker_segments", [])
            
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
        """
        try:
            print("Generating immediate response, bypassing cooldown...")
            old_cooldown = self.last_suggestion_time
            
            self.last_suggestion_time = 0
            
            suggestion = self.generate_suggestion(state, speaker)
            
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
            
        return conversation[-min(len(conversation), max_tokens*4):]
        
    def _get_latest_segments(self, segments: List[Dict[str, Any]], num_segments: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent speaker segments."""
        if not segments:
            return []
            
        return segments[-min(len(segments), num_segments):] 
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of the given text using VADER.
        """
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            
            try:
                analyzer = SentimentIntensityAnalyzer()
            except:
                import nltk
                nltk.download('vader_lexicon')
                analyzer = SentimentIntensityAnalyzer()
            
            scores = analyzer.polarity_scores(text)
            
            if scores['compound'] >= 0.05:
                sentiment = "positive"
            elif scores['compound'] <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            if sentiment == "positive":
                if scores['pos'] > 0.5:
                    emotion = "happy"
                else:
                    emotion = "content"
            elif sentiment == "negative":
                if scores['neg'] > 0.5:
                    emotion = "angry"
                else:
                    emotion = "sad"
            else:
                emotion = "neutral"
            
            sentiment_data = {
                "sentiment": sentiment,
                "confidence": abs(scores['compound']),
                "emotion": emotion,
                "scores": {
                    "positive": scores['pos'],
                    "negative": scores['neg'],
                    "neutral": scores['neu'],
                    "compound": scores['compound']
                }
            }
            
            print(f"Sentiment analysis completed: {sentiment_data}")
            return sentiment_data
            
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            import traceback
            traceback.print_exc()
            return {"sentiment": "neutral", "confidence": 0.5, "emotion": "unknown"}

    def _check_knowledge_base(self) -> None:
        """
        Check if a knowledge base exists for the current user and if its size
        exceeds the threshold. If under threshold, load into knowledge_base_content
        for direct inclusion in the prompt.
        """
        user_id = self._get_current_user_id()
        if not user_id:
            print("No user ID found. Cannot load knowledge base.")
            self.knowledge_base_content = None
            return
        
        kb_path = os.path.join(
            os.getcwd(),
            "conversations", 
            f"conversation_{user_id}", 
            "knowledge_base.txt"
        )
        
        if not os.path.exists(kb_path):
            print(f"No knowledge base found at {kb_path}")
            self.knowledge_base_content = None
            return
        
        try:
            with open(kb_path, 'r') as f:
                content = f.read()
            
            kb_size = len(content)
            print(f"Knowledge base size: {kb_size} characters")
            
            if kb_size <= self.knowledge_base_size_threshold:
                print(f"Knowledge base under {self.knowledge_base_size_threshold} characters. Including in prompt directly.")
                self.knowledge_base_content = content
            else:
                print(f"Knowledge base exceeds {self.knowledge_base_size_threshold} characters. Too large to include directly.")
                self.knowledge_base_content = None
                
        except Exception as e:
            print(f"Error checking knowledge base: {e}")
            self.knowledge_base_content = None
    
    def _get_current_user_id(self) -> Optional[str]:
        """
        Get the current user ID from current_user_id.txt.
        Returns None if no valid user ID exists.
        """
        user_id_file = os.path.join(os.getcwd(), "current_user_id.txt")
        
        if not os.path.exists(user_id_file):
            print("current_user_id.txt file not found")
            return None
        
        try:
            with open(user_id_file, 'r') as f:
                content = f.read().strip()
            
            if content == "NO_FACE_DETECTED" or not content:
                print("No face detected, no user ID available")
                return None
            
            if content.startswith("skip_upload:"):
                user_id = content.split("skip_upload:")[1]
            else:
                user_id = content
            
            print(f"Current user ID: {user_id}")
            return user_id
            
        except Exception as e:
            print(f"Error reading current user ID: {e}")
            return None