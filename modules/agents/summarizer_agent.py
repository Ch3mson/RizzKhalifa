#!/usr/bin/env python3

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from typing import Dict, List, Any

from modules.config import GROQ_MODEL

SUMMARIZER_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert summarizer. Summarize the following conversation in a short paragraph:

Conversation:
{conversation}

{speaker_info}

Return a concise summary as plain text that includes who said what."""
)

class SummarizerAgent:
    """Agent that summarizes conversations."""
    
    def __init__(self, model=GROQ_MODEL, temperature=0.2):
        self.llm = ChatGroq(model=model, temperature=temperature)
        self.parser = StrOutputParser()
        
        # Build chain
        self.chain = (
            RunnablePassthrough()
            | SUMMARIZER_PROMPT
            | self.llm
            | self.parser
        )
    
    def summarize(self, conversation: str, speaker_segments: List[Dict[str, Any]] = None) -> str:
        """
        Generate a summary of the conversation.
        """
        speaker_info = ""
        if speaker_segments:
            speakers = {}
            for segment in speaker_segments:
                speaker_id = segment.get("speaker", "UNKNOWN")
                person_name = segment.get("person", speaker_id)
                
                if speaker_id not in speakers:
                    speakers[speaker_id] = person_name
            
            if speakers:
                speaker_info = "Speaker Information:\n"
                for speaker_id, person_name in speakers.items():
                    speaker_info += f"- {speaker_id} is {person_name}\n"
        
        return self.chain.invoke({
            "conversation": conversation,
            "speaker_info": speaker_info
        }) 