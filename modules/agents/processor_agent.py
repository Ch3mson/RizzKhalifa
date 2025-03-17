#!/usr/bin/env python3

from typing import Dict, Any, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

from modules.config import GROQ_MODEL
from langchain_groq import ChatGroq
PROCESSOR_PROMPT = ChatPromptTemplate.from_template(
    """Analyze the following user statement and determine how it should be processed.

User statement:
{input_text}

Categorize this statement into one of these types:
1. SKIP - Meaningless dialogue like greetings, small talk, or filler (e.g., "hello", "how are you", "um", "hmm")
2. PERSONAL_INFO - Personal information that should be stored but NOT searched (e.g., "my birthday is July 10th", "I live in Boston")
3. SEARCH_TOPIC - ONLY use for explicit statements that fit into these categories:
   - Likes/Favorites: Food, music, movies, books, etc. SPECIFICALLY NOT GENERAL.
   - Dislikes: Things they explicitly don't like
   - School/Education: Subjects studying, degrees, schools
   - Hobbies/Activities: Sports, games, arts, etc.
   - Professional: Job, career field, work interests

IMPORTANT: Only categorize as SEARCH_TOPIC if there is an explicit statement that fits into one of the above categories, like:
- "I love/like/enjoy [food/music/etc.] this MUST be something specific and not general."
- "I study/major in [subject]"
- "My hobby is [activity]"
- "I work as/in [profession]"

Do NOT categorize general statements or discussions as SEARCH_TOPIC.

Respond with a JSON object containing:
- category: The category (SKIP, PERSONAL_INFO, or SEARCH_TOPIC)
- explanation: Brief explanation of why you categorized it this way 
- topics: If SEARCH_TOPIC, include:
  - name: The specific thing (e.g., "Sushi", "Computer Science")
  - category: One of [Likes, Dislikes, School, Hobbies, Professional, Skills]
  - description: Brief context about their relationship to it

Example responses:

For "I love eating sushi":
{{{{
  "category": "SEARCH_TOPIC",
  "explanation": "Explicit statement of food preference",
  "topics": [{{
    "name": "Sushi",
    "category": "Likes",
    "description": "Enjoys eating"
  }}]
}}}}

For "I hate classical music":
{{{{
  "category": "SEARCH_TOPIC",
  "explanation": "Explicit statement of music dislike",
  "topics": [{{
    "name": "Classical Music",
    "category": "Dislikes",
    "description": "Does not enjoy"
  }}]
}}}}

For "I'm studying computer science":
{{{{
  "category": "SEARCH_TOPIC",
  "explanation": "Explicit statement of educational focus",
  "topics": [{{
    "name": "Computer Science",
    "category": "School",
    "description": "Current field of study"
  }}]
}}}}
"""
)

class ProcessorAgent:
    """
    Agent that analyzes speech to determine:
    1. If the input is important enough to process
    2. Whether to perform a search on it
    3. What specific topics to search for
    """
    
    def __init__(self, model=GROQ_MODEL, temperature=0.1):
        self.llm = ChatGroq(model=model, temperature=temperature)
        self.parser = JsonOutputParser()
        
        self.chain = PROCESSOR_PROMPT | self.llm | self.parser
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Analyze the input text and determine its processing category.
        
        Returns:
            Dict with keys:
            - category: "SKIP", "PERSONAL_INFO", or "SEARCH_TOPIC"
            - explanation: Why it was categorized this way
            - topics: List of search topics (if applicable)
        """
        try:
            # print(f"Processing text input: {text[:50]}..." if len(text) > 50 else text)
            
            input_dict = {"input_text": text}
            
            result = self.chain.invoke(input_dict)
            
            # print(f"Processor result: category={result.get('category', 'UNKNOWN')}, topics={result.get('topics', [])}")
            
            if "category" not in result:
                result["category"] = "SKIP"
            if "explanation" not in result:
                result["explanation"] = "Failed to analyze content"
            if "topics" not in result:
                result["topics"] = []
                
            return result
        except Exception as e:
            print(f"Error in processor agent: {e}")
            import traceback
            traceback.print_exc()
            return {
                "category": "SKIP",
                "explanation": f"Error processing input: {str(e)}",
                "topics": []
            }
    
    def should_process(self, text: str) -> bool:
        """Helper method to determine if text should be processed at all."""
        result = self.process(text)
        return result["category"] != "SKIP"
    
    def should_search(self, text: str) -> Tuple[bool, list]:
        """
        Helper method to determine if search should be performed.
        """
        result = self.process(text)
        search_needed = result["category"] == "SEARCH_TOPIC"
        topics = result.get("topics", [])
        
        print(f"Should search: {search_needed}, Topics: {topics}")
        
        return (search_needed, topics)