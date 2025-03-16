#!/usr/bin/env python3

import os
import logging
import re
import time
from typing import Dict, Any, Optional, List, Tuple

# Import our RAG Manager
from modules.rag_manager import RAGManager

# Initialize logging
logger = logging.getLogger("rizz_rag_integration")

# Singleton RAG Manager instance
_rag_manager = None

# Simple result cache to avoid repeated searches
_result_cache = {}
_cache_expiry = 300  # 5 minutes cache validity

def get_rag_manager() -> RAGManager:
    """
    Get or create the RAG Manager singleton instance.
    Also preloads the vectorstore for faster initial performance.
    
    Returns:
        RAGManager: The singleton RAG Manager instance
    """
    global _rag_manager
    
    if _rag_manager is None:
        _rag_manager = RAGManager()
        # Preload the vectorstore for faster first use
        try:
            _rag_manager.preload_vectorstore()
        except Exception as e:
            logger.error(f"Error preloading vectorstore: {e}")
    
    return _rag_manager

def _cache_key(message: str, history: List[str] = None) -> str:
    """
    Generate a cache key from the message and conversation history.
    
    Args:
        message: User message
        history: Conversation history
        
    Returns:
        str: Cache key
    """
    history_hash = "none" if not history else str(hash(tuple(history[-3:] if len(history) > 3 else history)))
    return f"{message[:100]}:{history_hash}"

def enhance_prompt_with_rag(message: str, conversation_history: List[str] = None, min_similarity: float = 0.6) -> str:
    """
    Enhance a user message with relevant context from the knowledge base
    using RAG with conversation history.
    
    Args:
        message: The user message to enhance
        conversation_history: Previous exchanges in the conversation
        min_similarity: Minimum similarity score threshold
        
    Returns:
        str: Context from the knowledge base that can be included in the prompt
    """
    global _result_cache
    
    # Check cache first for faster responses
    current_time = time.time()
    cache_key = _cache_key(message, conversation_history)
    
    if cache_key in _result_cache:
        cache_entry = _result_cache[cache_key]
        if current_time - cache_entry["timestamp"] < _cache_expiry:
            logger.info(f"Using cached RAG result for: {message[:50]}...")
            return cache_entry["result"]
    
    # Not in cache, perform the search
    rag = get_rag_manager()
    context = rag.get_rag_context(
        message=message,
        conversation_history=conversation_history, 
        min_similarity=min_similarity
    )
    
    # Cache the result
    _result_cache[cache_key] = {
        "result": context,
        "timestamp": current_time
    }
    
    # Prune cache if it gets too large (keep most recent 100 entries)
    if len(_result_cache) > 100:
        oldest_keys = sorted(_result_cache.keys(), 
                            key=lambda k: _result_cache[k]["timestamp"])[:50]
        for key in oldest_keys:
            del _result_cache[key]
    
    return context

def extract_conversation_history(agent_input: Dict[str, Any]) -> List[str]:
    """
    Extract conversation history from the agent input.
    
    Args:
        agent_input: The agent input dictionary
        
    Returns:
        List[str]: Extracted conversation messages
    """
    history = []
    
    # Check if we have conversation history in the input
    if 'conversation_history' in agent_input and isinstance(agent_input['conversation_history'], list):
        # Direct use of provided conversation history
        history = agent_input['conversation_history']
    
    # Check if we have a conversation string that needs parsing
    elif 'conversation' in agent_input and isinstance(agent_input['conversation'], str):
        # Try to extract messages from conversation string
        conversation = agent_input['conversation']
        
        # Simple pattern matching to extract messages
        # This is a basic example - adjust based on your conversation format
        user_messages = re.findall(r'User: (.*?)(?=\n|$)', conversation)
        ai_messages = re.findall(r'AI: (.*?)(?=\n|$)', conversation)
        
        # Interleave messages to maintain conversation flow
        for i in range(max(len(user_messages), len(ai_messages))):
            if i < len(user_messages):
                history.append(user_messages[i])
            if i < len(ai_messages):
                history.append(ai_messages[i])
    
    # Add the current query if available
    if 'user_query' in agent_input and agent_input['user_query']:
        history.append(agent_input['user_query'])
    
    return history

def rizz_cursor_rag_middleware(agent_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Middleware function to process the input to the rizz_cursor_agent.
    This adds relevant context from the knowledge base based on the user's query
    and conversation history.
    
    Args:
        agent_input: The input dictionary for the agent
        
    Returns:
        Dict[str, Any]: Enhanced input with RAG context
    """
    # Check if we should process this agent
    agent_name = agent_input.get('agent_name', '')
    if agent_name != 'rizz_cursor_agent':
        return agent_input
    
    # Skip middleware if agent_input already has the rag_processed flag
    if agent_input.get('rag_processed', False):
        return agent_input
    
    try:
        # Extract the user query
        user_query = agent_input.get('user_query', '')
        if not user_query:
            return agent_input
        
        # Extract conversation history
        conversation_history = extract_conversation_history(agent_input)
        
        # Log the conversation history being used
        if conversation_history:
            logger.info(f"Using conversation history with {len(conversation_history)} messages")
            logger.debug(f"Conversation history: {conversation_history[-3:] if len(conversation_history) > 3 else conversation_history}")
        
        # Get RAG context with conversation history
        rag_context = enhance_prompt_with_rag(
            message=user_query,
            conversation_history=conversation_history
        )
        
        # If we have meaningful RAG context, add it to the input
        if rag_context and rag_context not in ["No user identified.", 
                                              "No knowledge base found for this user.",
                                              "No relevant information found in knowledge base."]:
            # Modify the agent input to include RAG context
            if 'additional_context' not in agent_input:
                agent_input['additional_context'] = ''
            
            # Add the RAG context to the additional context
            agent_input['additional_context'] += f"\n\n{rag_context}"
            
            # Log that we enhanced the prompt
            logger.info(f"Enhanced prompt with RAG context for user query: {user_query[:50]}...")
        
        # Mark as processed to avoid duplicate processing
        agent_input['rag_processed'] = True
        
        return agent_input
    
    except Exception as e:
        logger.error(f"Error in rizz_cursor_rag_middleware: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Return original input in case of any errors
        return agent_input

def is_knowledge_base_available() -> bool:
    """
    Check if a knowledge base is available for the current user.
    
    Returns:
        bool: True if a knowledge base is available, False otherwise
    """
    rag = get_rag_manager()
    user_id = rag.get_current_user_id()
    
    if not user_id:
        return False
    
    # Check if the knowledge base file exists
    kb_path = os.path.join(
        rag.conversations_dir,
        f"conversation_{user_id}",
        "knowledge_base.txt"
    )
    
    return os.path.exists(kb_path)

if __name__ == "__main__":
    # Simple test
    test_input = {
        "agent_name": "rizz_cursor_agent",
        "user_query": "Tell me about my favorite food and restaurants",
        "conversation": """
        User: Hi there! I'm thinking about going out to eat tonight.
        AI: Hello! That sounds like a great idea. Any specific type of cuisine you're in the mood for?
        User: I'm not sure yet, what would you recommend?
        """
    }
    
    # Try with conversation history
    enhanced_input = rizz_cursor_rag_middleware(test_input)
    
    print("Original input:")
    print(test_input)
    print("\nEnhanced input:")
    for key, value in enhanced_input.items():
        if key == 'additional_context' and value:
            print(f"\n{key}:")
            print(value)
        else:
            print(f"{key}: {value}") 