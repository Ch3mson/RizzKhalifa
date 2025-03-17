#!/usr/bin/env python3

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from modules.config import OPENAI_MODEL, GROQ_MODEL


KNOWLEDGE_RESPONSE_PROMPT = ChatPromptTemplate.from_template(
    """Use the conversation snippet below and the knowledge base to form an informative, helpful reply.

Recent snippet: {recent_conversation}

Relevant knowledge (current 2024-2025 information to provide detailed, accurate responses):
{knowledge_base}

Give an informative reply that:
1. References specific facts from the knowledge base when relevant
2. Incorporates the user's interests or personal details where appropriate
3. Provides substantive, helpful information rather than generic responses
4. Stays conversational in tone while being educational and informative
5. Emphasizes the currency of information (from 2024-2025) when appropriate
"""
)

class ResponseGenerationAgent:
    """Agent that generates responses based on conversation and knowledge base."""
    
    def __init__(self, model=GROQ_MODEL, temperature=0.7):
        self.llm = ChatGroq(model=model, temperature=temperature)
        self.parser = StrOutputParser()
        
        # Build chain
        self.chain = (
            {
                "recent_conversation": RunnablePassthrough(),
                "knowledge_base": RunnablePassthrough(),
            }
            | KNOWLEDGE_RESPONSE_PROMPT
            | self.llm
            | self.parser
        )
    
    def generate_response(self, recent_conversation: str, knowledge_base: str) -> str:
        """Generate a response based on recent conversation and knowledge base."""
        return self.chain.invoke({
            "recent_conversation": recent_conversation,
            "knowledge_base": knowledge_base
        }) 