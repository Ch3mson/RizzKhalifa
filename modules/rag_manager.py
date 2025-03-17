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
    """
    
    def __init__(self):
        self.logger = logging.getLogger("rag_manager")
        
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimensions = 1536 
        
        self.project_dir = os.getcwd()
        self.conversations_dir = os.path.join(self.project_dir, "conversations")
        self.cache_dir = os.path.join(self.project_dir, "conversations", "system_data", "embedding_cache")
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.chunk_size = 1000 
        self.chunk_overlap = 200  
        
        self.current_user_id = None
        self.user_kb_path = None
        self.user_kb_content = None
        self.vectorstore = None 
        
        self.vectorstore_cache = {}
        
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
        """
        user_id = self.get_current_user_id()
        if not user_id:
            print("No user ID found. Cannot preload vectorstore.")
            return
        
        if user_id in self.vectorstore_cache:
            try:
                cache_entry = self.vectorstore_cache[user_id]
                self.vectorstore = cache_entry.get('vectorstore')
                print(f"Preloaded vectorstore for user {user_id} from cache")
                
                self.current_user_id = user_id
                self.user_kb_path = os.path.join(
                    self.conversations_dir, 
                    f"conversation_{user_id}", 
                    "knowledge_base.txt"
                )
                
                self.embeddings.embed_query("Test query to warm up the embedding model")
                print("Embedding model initialized")
                
                return
            except Exception as e:
                print(f"Error preloading vectorstore: {e}")
        
        self.load_knowledge_base(user_id)
    
    def get_current_user_id(self) -> Optional[str]:
        """
        Get the current user ID from current_user_id.txt.
        Returns None if no valid user ID exists.
        """
        user_id_file = os.path.join(self.project_dir, "current_user_id.txt")
        
        if not os.path.exists(user_id_file):
            return None
        
        try:
            with open(user_id_file, 'r') as f:
                content = f.read().strip()
            
            if content == "NO_FACE_DETECTED" or not content:
                return None
            
            if content.startswith("skip_upload:"):
                user_id = content.split("skip_upload:")[1]
            else:
                user_id = content
            
            print(f"Current user ID: {user_id}")
            return user_id
            
        except Exception as e:
            self.logger.error(f"Error reading current user ID: {e}")
            return None
    
    def load_knowledge_base(self, user_id: str = None) -> bool:
        """
        Load the knowledge base for a specific user ID or the current user.
        """
        self.current_user_id = user_id
        
        self.user_kb_path = os.path.join(
            self.conversations_dir, 
            f"conversation_{user_id}", 
            "knowledge_base.txt"
        )
        
        if not os.path.exists(self.user_kb_path):
            print(f"Knowledge base not found at: {self.user_kb_path}")
            return False
        
        try:
            with open(self.user_kb_path, 'r') as f:
                self.user_kb_content = f.read()
            
            print(f"Loaded knowledge base for user {user_id} ({len(self.user_kb_content)} bytes)")
            
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
        
        if self.current_user_id in self.vectorstore_cache:
            cache_entry = self.vectorstore_cache[self.current_user_id]
            
            try:
                file_mtime = os.path.getmtime(self.user_kb_path)
                cache_time = cache_entry.get('timestamp', 0)
                
                if cache_time > file_mtime:
                    self.vectorstore = cache_entry.get('vectorstore')
                    return
            except Exception as e:
                print(f"Error checking cache validity: {e}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(self.user_kb_content)
        
        documents = [Document(page_content=chunk) for chunk in chunks]
        
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        self.vectorstore_cache[self.current_user_id] = {
            'vectorstore': self.vectorstore,
            'timestamp': time.time()
        }
        
        self._save_vectorstore_cache()
        
    def search_knowledge_base(self, query: str, top_k: int = 3, history: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for chunks most relevant to the query and conversation history.
        """
        if not self.vectorstore:
            print("Knowledge base not loaded. Call load_knowledge_base() first.")
            return []
        
        try:
            enhanced_query = query
            
            if history and len(history) > 0:
                last_messages = history[-3:] if len(history) > 3 else history
                history_context = " ".join(last_messages)
                
                enhanced_query = f"{history_context} {query}"
                print(f"Enhanced query with conversation history: {enhanced_query[:100]}...")
            
            embeddings_filter = EmbeddingsFilter(
                embeddings=self.embeddings,
                similarity_threshold=0.6 
            )
            
            retriever = ContextualCompressionRetriever(
                base_compressor=embeddings_filter,
                base_retriever=self.vectorstore.as_retriever(search_kwargs={"k": top_k + 2})  # Get a few extras to filter
            )
            
            docs = retriever.get_relevant_documents(enhanced_query)
            
            results = []
            for i, doc in enumerate(docs[:top_k]):
                doc_embedding = self.embeddings.embed_query(doc.page_content)
                query_embedding = self.embeddings.embed_query(query)
                
                similarity = self._calculate_similarity(query_embedding, doc_embedding)
                
                results.append({
                    'text': doc.page_content,
                    'score': similarity
                })
            
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {e}")
            print(f"Error searching knowledge base: {e}")
            return []
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        vec1 = np.array(embedding1).reshape(1, -1)
        vec2 = np.array(embedding2).reshape(1, -1)
        
        dot_product = np.dot(vec1, vec2.T)[0][0]
        
        mag1 = np.sqrt(np.sum(vec1**2))
        mag2 = np.sqrt(np.sum(vec2**2))
        
        if mag1 * mag2 == 0:
            return 0
        return dot_product / (mag1 * mag2)
    
    def get_rag_context(self, message: str, conversation_history: List[str] = None, min_similarity: float = 0.6) -> str:
        """
        Get RAG context for the rizz_cursor_agent based on the current message and conversation history.
        """
        user_id = self.get_current_user_id()
        if not user_id:
            return "No user identified."
        
        if self.current_user_id != user_id or not self.vectorstore:
            success = self.load_knowledge_base(user_id)
            if not success:
                return "No knowledge base found for this user."
        
        results = self.search_knowledge_base(message, top_k=5, history=conversation_history)
        
        filtered_results = [r for r in results if r['score'] >= min_similarity]
        
        if not filtered_results:
            return "No relevant information found in knowledge base."
        
        context = "### Relevant information from user's knowledge base:\n\n"
        
        for i, result in enumerate(filtered_results, 1):
            context += f"[Match {i} - {result['score']:.2f}] {result['text']}\n\n"
        
        return context
