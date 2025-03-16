#!/usr/bin/env python3

import os
import sys
from modules.rag_manager import RAGManager
from modules.rizz_rag_integration import enhance_prompt_with_rag, rizz_cursor_rag_middleware, extract_conversation_history

def test_knowledge_base_lookup():
    """Test the RAG system's ability to find relevant information in the knowledge base."""
    # Create a RAG manager instance
    rag = RAGManager()
    
    # Get the current user ID
    user_id = rag.get_current_user_id()
    print(f"Current user ID: {user_id}")
    
    # Load the knowledge base
    if user_id:
        success = rag.load_knowledge_base(user_id)
        if success:
            print(f"Successfully loaded knowledge base for user {user_id}")
        else:
            print(f"Failed to load knowledge base for user {user_id}")
            return
    else:
        print("No user ID found")
        return
    
    # Test basic queries without conversation history
    print("\n\n" + "="*50)
    print("BASIC QUERIES WITHOUT CONVERSATION HISTORY")
    print("="*50)
    
    test_queries = [
        "What are my hobbies?",
        "What food do I like?",
        "Where did I go to school?",
        "What are my career goals?",
        "Tell me about my travel experiences",
        "What kind of music do I enjoy?"
    ]
    
    # Run tests for each query
    for query in test_queries:
        print(f"\n\n{'='*50}")
        print(f"QUERY: {query}")
        print(f"{'='*50}")
        
        # Search the knowledge base
        results = rag.search_knowledge_base(query, top_k=2)
        
        # Print results
        if results:
            print(f"Found {len(results)} relevant chunks:")
            for i, result in enumerate(results, 1):
                print(f"\nResult {i} (score: {result['score']:.4f}):")
                print(f"{result['text']}")
        else:
            print("No relevant information found")
    
    # Test queries with conversation history
    print("\n\n" + "="*50)
    print("QUERIES WITH CONVERSATION HISTORY")
    print("="*50)
    
    conversation_scenarios = [
        {
            "history": [
                "I'm thinking about taking up a new hobby",
                "Something that gets me outdoors"
            ],
            "query": "What activities do I currently enjoy?"
        },
        {
            "history": [
                "I'm hungry and want to eat out tonight",
                "I'm in the mood for something spicy"
            ],
            "query": "What restaurants do I like?"
        },
        {
            "history": [
                "I'm planning my next vacation",
                "I'd like to go somewhere I haven't been before"
            ],
            "query": "Where have I traveled to recently?"
        }
    ]
    
    for scenario in conversation_scenarios:
        print(f"\n\n{'='*50}")
        print(f"CONVERSATION HISTORY: {scenario['history']}")
        print(f"QUERY: {scenario['query']}")
        print(f"{'='*50}")
        
        # Search the knowledge base with history
        results = rag.search_knowledge_base(scenario['query'], top_k=2, history=scenario['history'])
        
        # Print results
        if results:
            print(f"Found {len(results)} relevant chunks:")
            for i, result in enumerate(results, 1):
                print(f"\nResult {i} (score: {result['score']:.4f}):")
                print(f"{result['text']}")
        else:
            print("No relevant information found")
    
    # Test the middleware function with a simulated conversation
    print("\n\n" + "="*50)
    print("TESTING MIDDLEWARE FUNCTION WITH CONVERSATION")
    print("="*50)
    
    test_input = {
        "agent_name": "rizz_cursor_agent",
        "user_query": "Tell me about my favorite food and restaurants",
        "conversation": """
        User: Hi there! I'm thinking about going out to eat tonight.
        AI: Hello! That sounds like a great idea. Any specific type of cuisine you're in the mood for?
        User: I'm not sure yet, what would you recommend based on my usual preferences?
        """
    }
    
    # Test conversation history extraction
    conversation_history = extract_conversation_history(test_input)
    print("\nExtracted conversation history:")
    for i, msg in enumerate(conversation_history):
        print(f"{i+1}: {msg}")
    
    # Test the middleware function
    enhanced_input = rizz_cursor_rag_middleware(test_input)
    
    print("\nOriginal input:")
    print(f"agent_name: {test_input['agent_name']}")
    print(f"user_query: {test_input['user_query']}")
    print(f"conversation: (length {len(test_input['conversation'])} chars)")
    
    print("\nEnhanced input:")
    for key, value in enhanced_input.items():
        if key == 'additional_context' and value:
            print(f"\n{key}:")
            print(value)
        elif key == 'conversation':
            print(f"{key}: (length {len(value)} chars)")
        else:
            print(f"{key}: {value}")
    
    print("\n\n" + "="*50)
    print("Test using direct conversation_history list")
    print("="*50)
    
    test_input_direct = {
        "agent_name": "rizz_cursor_agent",
        "user_query": "What kind of Asian food do I like?",
        "conversation_history": [
            "I love cooking",
            "Asian cuisine is my favorite to make at home",
            "I especially enjoy spicy dishes"
        ]
    }
    
    enhanced_input_direct = rizz_cursor_rag_middleware(test_input_direct)
    
    print("\nOriginal input:")
    print(f"agent_name: {test_input_direct['agent_name']}")
    print(f"user_query: {test_input_direct['user_query']}")
    print("conversation_history:")
    for msg in test_input_direct['conversation_history']:
        print(f"  - {msg}")
    
    print("\nEnhanced input:")
    for key, value in enhanced_input_direct.items():
        if key == 'additional_context' and value:
            print(f"\n{key}:")
            print(value)
        elif key == 'conversation_history':
            print(f"{key}: (list with {len(value)} items)")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    test_knowledge_base_lookup() 