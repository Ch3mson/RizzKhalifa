#!/usr/bin/env python3

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

from modules.config import OPENAI_MODEL

PERSONAL_INFO_PROMPT = ChatPromptTemplate.from_template(
    """Given the conversation summary below, identify any personal information the user shared 
(e.g. "likes sushi", "birthday is July 10", "dislikes pineapple", etc.). 
For each piece of info, specify "type" (what kind of info), "value" (user statement), and "confidence" (0-1 scale).

Conversation summary:
{summary}

Return JSON with a "personal_details" array, 
each containing "type", "value", and "confidence"."""
)

class PersonalInfoAgent:
    """Agent that extracts personal information from conversation summaries."""
    
    def __init__(self, model=OPENAI_MODEL, temperature=0.1):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.parser = JsonOutputParser()
        
        # Build chain
        self.chain = (
            {"summary": RunnablePassthrough()}
            | PERSONAL_INFO_PROMPT
            | self.llm
            | self.parser
        )
    
    def extract_personal_info(self, summary: str) -> dict:
        """Extract personal information from a conversation summary."""
        result = self.chain.invoke({"summary": summary})
        return result.get("personal_details", []) 