#!/usr/bin/env python3

import os
import json
import time
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import pickle
from pathlib import Path

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

class RAGManager:
    """
    A module for managing Retrieval-Augmented Generation based on user IDs.
    This module:
    1. Gets the current user ID from current_user_id.txt
    2. Loads the knowledge base for that user 
    3. Uses LangChain with OpenAI's text-embedding-3-small for generating embeddings
    4. Performs similarity search to retrieve relevant information
    5. Caches embeddings locally for faster retrieval
    6. Incorporates conversation history for better context
    """
    
    def __init__(self):
        # Set up logging
        self.logger = logging.getLogger("rag_manager")
        
        # Initialize LangChain embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Embedding model settings
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimensions = 1536  # Dimensions for text-embedding-3-small
        
        # Set up paths
        self.project_dir = os.getcwd()
        self.conversations_dir = os.path.join(self.project_dir, "conversations")
        self.cache_dir = os.path.join(self.project_dir, "conversations", "system_data", "embedding_cache")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Store settings
        self.chunk_size = 1000  # LangChain character chunk size
        self.chunk_overlap = 200  # LangChain chunk overlap
        
        # Current user state
        self.current_user_id = None
        self.user_kb_path = None
        self.user_kb_content = None
        self.vectorstore = None  # LangChain vectorstore
        
        # Cache of vectorstores by user ID
        self.vectorstore_cache = {}
        
        # Try to load existing vectorstore cache
        self._load_vectorstore_cache()
        
    def _load_vectorstore_cache(self) -> None:
        """Load the vectorstore cache from disk if it exists."""
        cache_file = os.path.join(self.cache_dir, "vectorstore_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.vectorstore_cache = pickle.load(f)
                self.logger.info(f"Loaded vectorstore cache with {len(self.vectorstore_cache)} entries")
                print(f"Loaded vectorstore cache with {len(self.vectorstore_cache)} entries")
            except Exception as e:
                self.logger.error(f"Error loading vectorstore cache: {e}")
                print(f"Error loading vectorstore cache: {e}")
                self.vectorstore_cache = {}
    
    def _save_vectorstore_cache(self) -> None:
        """Save the vectorstore cache to disk."""
        cache_file = os.path.join(self.cache_dir, "vectorstore_cache.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.vectorstore_cache, f)
            self.logger.info(f"Saved vectorstore cache with {len(self.vectorstore_cache)} entries")
        except Exception as e:
            self.logger.error(f"Error saving vectorstore cache: {e}")
            print(f"Error saving vectorstore cache: {e}")
    
    def preload_vectorstore(self) -> None:
        """
        Preload the vectorstore for faster initial performance.
        This method should be called during application startup to
        avoid delays when first using the RAG system.
        """
        user_id = self.get_current_user_id()
        if not user_id:
            print("No user ID found. Cannot preload vectorstore.")
            return
        
        # If the vectorstore is already in cache, load it
        if user_id in self.vectorstore_cache:
            try:
                cache_entry = self.vectorstore_cache[user_id]
                self.vectorstore = cache_entry.get('vectorstore')
                print(f"Preloaded vectorstore for user {user_id} from cache")
                
                # Set the current user ID and path
                self.current_user_id = user_id
                self.user_kb_path = os.path.join(
                    self.conversations_dir, 
                    f"conversation_{user_id}", 
                    "knowledge_base.txt"
                )
                
                # Also run a test embedding to ensure the model is loaded
                self.embeddings.embed_query("Test query to warm up the embedding model")
                print("Embedding model initialized")
                
                return
            except Exception as e:
                print(f"Error preloading vectorstore: {e}")
        
        # If not in cache, load the knowledge base the normal way
        self.load_knowledge_base(user_id)
    
    def get_current_user_id(self) -> Optional[str]:
        """
        Get the current user ID from current_user_id.txt.
        Returns None if no valid user ID exists.
        """
        user_id_file = os.path.join(self.project_dir, "current_user_id.txt")
        
        if not os.path.exists(user_id_file):
            print("current_user_id.txt file not found")
            return None
        
        try:
            with open(user_id_file, 'r') as f:
                content = f.read().strip()
            
            # Check special cases
            if content == "NO_FACE_DETECTED" or not content:
                print("No face detected, no user ID available")
                return None
            
            # Handle existing face format
            if content.startswith("skip_upload:"):
                user_id = content.split("skip_upload:")[1]
            else:
                user_id = content
            
            print(f"Current user ID: {user_id}")
            return user_id
            
        except Exception as e:
            self.logger.error(f"Error reading current user ID: {e}")
            print(f"Error reading current user ID: {e}")
            return None
    
    def load_knowledge_base(self, user_id: str = None) -> bool:
        """
        Load the knowledge base for a specific user ID or the current user.
        
        Args:
            user_id: Optional user ID to load. If None, uses the current user ID.
            
        Returns:
            bool: True if knowledge base was loaded successfully, False otherwise.
        """
        # Get the current user ID if not provided
        if user_id is None:
            user_id = self.get_current_user_id()
            if user_id is None:
                return False
        
        # Store the current user ID
        self.current_user_id = user_id
        
        # Define the knowledge base path
        self.user_kb_path = os.path.join(
            self.conversations_dir, 
            f"conversation_{user_id}", 
            "knowledge_base.txt"
        )
        
        # Check if the knowledge base exists
        if not os.path.exists(self.user_kb_path):
            print(f"Knowledge base not found at: {self.user_kb_path}")
            return False
        
        try:
            # Load the knowledge base content
            with open(self.user_kb_path, 'r') as f:
                self.user_kb_content = f.read()
            
            print(f"Loaded knowledge base for user {user_id} ({len(self.user_kb_content)} bytes)")
            
            # Process the knowledge base into vector store
            self._process_knowledge_base()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {e}")
            print(f"Error loading knowledge base: {e}")
            return False
    
    def _process_knowledge_base(self) -> None:
        """Process the knowledge base content into chunks and generate a vectorstore using LangChain."""
        if not self.user_kb_content:
            return
        
        # Check if we already have a vectorstore for this user
        if self.current_user_id in self.vectorstore_cache:
            # Check if the cache has a timestamp and validate it against file modification time
            cache_entry = self.vectorstore_cache[self.current_user_id]
            
            try:
                file_mtime = os.path.getmtime(self.user_kb_path)
                cache_time = cache_entry.get('timestamp', 0)
                
                # If the cache is newer than the file, use it
                if cache_time > file_mtime:
                    self.vectorstore = cache_entry.get('vectorstore')
                    print(f"Using cached vectorstore for user {self.current_user_id}")
                    return
            except Exception as e:
                print(f"Error checking cache validity: {e}")
        
        # Split the text into manageable chunks using LangChain
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(self.user_kb_content)
        print(f"Split knowledge base into {len(chunks)} chunks")
        
        # Create documents from chunks
        documents = [Document(page_content=chunk) for chunk in chunks]
        
        # Create a vectorstore from the documents
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Cache the vectorstore
        self.vectorstore_cache[self.current_user_id] = {
            'vectorstore': self.vectorstore,
            'timestamp': time.time()
        }
        
        # Save the updated cache
        self._save_vectorstore_cache()
        
    def search_knowledge_base(self, query: str, top_k: int = 3, history: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for chunks most relevant to the query and conversation history.
        
        Args:
            query: The search query
            top_k: Number of top matches to return
            history: Optional list of previous conversation messages
            
        Returns:
            List of dictionaries with 'text' and 'score' keys
        """
        # Check if vectorstore is loaded
        if not self.vectorstore:
            print("Knowledge base not loaded. Call load_knowledge_base() first.")
            return []
        
        try:
            # Create an enhanced query that includes context from the conversation history
            enhanced_query = query
            
            if history and len(history) > 0:
                # Only use the last few messages to keep context manageable
                last_messages = history[-3:] if len(history) > 3 else history
                history_context = " ".join(last_messages)
                
                # Combine the current query with context from history
                enhanced_query = f"{history_context} {query}"
                print(f"Enhanced query with conversation history: {enhanced_query[:100]}...")
            
            # Create embeddings filter for more relevant results
            embeddings_filter = EmbeddingsFilter(
                embeddings=self.embeddings,
                similarity_threshold=0.6  # Adjust threshold as needed
            )
            
            # Create a contextual compression retriever
            retriever = ContextualCompressionRetriever(
                base_compressor=embeddings_filter,
                base_retriever=self.vectorstore.as_retriever(search_kwargs={"k": top_k + 2})  # Get a few extras to filter
            )
            
            # Get results
            docs = retriever.get_relevant_documents(enhanced_query)
            
            # Process results into the expected format
            results = []
            for i, doc in enumerate(docs[:top_k]):
                # Estimate similarity score - not directly available from compressed retriever
                # This is a workaround that approximates the score
                doc_embedding = self.embeddings.embed_query(doc.page_content)
                query_embedding = self.embeddings.embed_query(query)
                
                # Calculate cosine similarity (simple approximation)
                similarity = self._calculate_similarity(query_embedding, doc_embedding)
                
                results.append({
                    'text': doc.page_content,
                    'score': similarity
                })
            
            # Sort by similarity
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {e}")
            print(f"Error searching knowledge base: {e}")
            return []
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        # Convert to numpy arrays
        vec1 = np.array(embedding1).reshape(1, -1)
        vec2 = np.array(embedding2).reshape(1, -1)
        
        # Calculate dot product
        dot_product = np.dot(vec1, vec2.T)[0][0]
        
        # Calculate magnitudes
        mag1 = np.sqrt(np.sum(vec1**2))
        mag2 = np.sqrt(np.sum(vec2**2))
        
        # Calculate cosine similarity
        if mag1 * mag2 == 0:
            return 0
        return dot_product / (mag1 * mag2)
    
    def get_rag_context(self, message: str, conversation_history: List[str] = None, min_similarity: float = 0.6) -> str:
        """
        Get RAG context for the rizz_cursor_agent based on the current message and conversation history.
        This function loads the knowledge base, searches it, and returns
        relevant context to be included in the prompt.
        
        Args:
            message: The user message to search context for
            conversation_history: Optional list of previous exchanges in the conversation
            min_similarity: Minimum similarity score to include in results
            
        Returns:
            str: Formatted context that can be included in the prompt
        """
        # Get current user ID and load knowledge base
        user_id = self.get_current_user_id()
        if not user_id:
            return "No user identified."
        
        # Load the knowledge base if needed
        if self.current_user_id != user_id or not self.vectorstore:
            success = self.load_knowledge_base(user_id)
            if not success:
                return "No knowledge base found for this user."
        
        # Search the knowledge base using conversation history
        results = self.search_knowledge_base(message, top_k=5, history=conversation_history)
        
        # Filter results by minimum similarity
        filtered_results = [r for r in results if r['score'] >= min_similarity]
        
        if not filtered_results:
            return "No relevant information found in knowledge base."
        
        # Format the results for the prompt
        context = "### Relevant information from user's knowledge base:\n\n"
        
        for i, result in enumerate(filtered_results, 1):
            # Add the text with confidence score
            context += f"[Match {i} - {result['score']:.2f}] {result['text']}\n\n"
        
        return context


if __name__ == "__main__":
    # Simple test
    rag = RAGManager()
    user_id = rag.get_current_user_id()
    
    if user_id:
        rag.load_knowledge_base(user_id)
        # Test with some conversation history
        history = [
            "I'm thinking about picking up a new hobby",
            "Something that gets me outdoors would be nice"
        ]
        results = rag.search_knowledge_base("What kinds of activities do I enjoy?", history=history)
        
        print("\nSearch results with conversation history:")
        for r in results:
            print(f"Score: {r['score']:.4f}")
            print(f"Text: {r['text'][:100]}...")
            print() 