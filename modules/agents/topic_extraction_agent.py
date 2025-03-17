#!/usr/bin/env python3

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from modules.config import OPENAI_MODEL, GROQ_MODEL


# Define the prompt template
TOPIC_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert conversation analyst with a focus on identifying and categorizing personal preferences and activities.

From the summary below, extract ONLY topics that fit into these specific categories:
1. Likes/Favorites: Food, music, movies, books, etc. they explicitly enjoy
2. Dislikes: Things they explicitly don't like or dislike
3. School/Education: Subjects they're studying, degrees, schools
4. Hobbies/Activities: Sports, games, arts, etc. they do for fun
5. Professional: Job, career field, work interests
6. Skills: Languages, technical abilities, etc. they possess

ONLY extract a topic if there is a clear statement like:
- "I love/like/enjoy [food/music/etc.]" → Likes category
- "I hate/dislike/can't stand [thing]" → Dislikes category
- "I study/major in [subject]" → School category
- "My hobby is [activity]" → Hobbies category
- "I work as/in [profession]" → Professional category
- "I know/speak [language/skill]" → Skills category

Do NOT extract general topics that are merely discussed without an explicit statement.

Conversation summary:
{summary}

Return JSON with a "topics" array, each item containing:
- name: The specific thing (e.g., "Sushi", "Computer Science")
- category: One of [Likes, Dislikes, School, Hobbies, Professional, Skills]
- description: Brief context about their relationship to it

If no explicit preferences are found, return an empty topics array."""
)

DIRECT_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert at identifying and categorizing personal preferences and activities mentioned in text.

From the text below, extract ONLY topics that fit into these specific categories:
1. Likes/Favorites: Food, music, movies, books, etc. they explicitly enjoy
2. Dislikes: Things they explicitly don't like or dislike
3. School/Education: Subjects they're studying, degrees, schools
4. Hobbies/Activities: Sports, games, arts, etc. they do for fun
5. Professional: Job, career field, work interests
6. Skills: Languages, technical abilities, etc. they possess

ONLY extract a topic if there is a clear statement like:
- "I love/like/enjoy [food/music/etc.]" → Likes category
- "I hate/dislike/can't stand [thing]" → Dislikes category
- "I study/major in [subject]" → School category
- "My hobby is [activity]" → Hobbies category
- "I work as/in [profession]" → Professional category
- "I know/speak [language/skill]" → Skills category

Do NOT extract general topics that are merely discussed without an explicit statement.

Text:
{text}

Return JSON with a "topics" array, each item containing:
- name: The specific thing (e.g., "Sushi", "Computer Science")
- category: One of [Likes, Dislikes, School, Hobbies, Professional, Skills]
- description: Brief context about their relationship to it

If no explicit preferences are found, return an empty topics array."""
)

class TopicExtractionAgent:
    """Agent that extracts topics from conversation summaries."""
    
    def __init__(self, model=GROQ_MODEL, temperature=0.2):
        self.llm = ChatGroq(model=model, temperature=temperature)
        self.parser = JsonOutputParser()
        
        self.summary_chain = (
            {"summary": RunnablePassthrough()}
            | TOPIC_EXTRACTION_PROMPT
            | self.llm
            | self.parser
        )
        
        self.direct_chain = (
            {"text": RunnablePassthrough()}
            | DIRECT_EXTRACTION_PROMPT
            | self.llm
            | self.parser
        )
    
    def extract_topics(self, summary: str) -> dict:
        """Extract discussion topics from a conversation summary."""
        result = self.summary_chain.invoke({"summary": summary})
        return result.get("topics", [])
        
    def extract_topics_from_text(self, text: str) -> dict:
        """
        Extract discussion topics directly from conversation text.
        This is useful when we don't have a summary yet.
        """
        try:
            result = self.direct_chain.invoke({"text": text})
            return result.get("topics", [])
        except Exception as e:
            print(f"Error extracting topics from text: {e}")
            return [] 