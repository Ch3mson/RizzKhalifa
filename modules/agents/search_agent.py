#!/usr/bin/env python3

from typing import List, Dict
from langchain_community.utilities import SearchApiAPIWrapper
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun

class SearchAgent:
    """
    Wraps the SearchApiAPIWrapper to get comprehensive content from websites.
    """
    def __init__(self):
        self.search = DuckDuckGoSearchRun()
        self.tool = Tool(
            name="web_search",
            description="Use web search to find comprehensive, detailed information.",
            func=self.search.run,
        )
    
    def search_topics(self, topics: List[str], num_results: int = 6) -> Dict[str, List[str]]:
        """
        For each topic, run a search and gathers info.
        Restricts results to 2024-2025 timeframe.
        """
        search_results = {}
        for topic in topics:
            topic = topic.strip()
            if not topic:
                continue
            try:
                response = self.search.run(
                    f"{topic} detailed information guide explanation after:2024-01-01 before:2025-12-31"
                )
                
                # Split the response into paragraphs
                paragraphs = response.split('\n\n')
                
                # Filter out empty paragraphs
                detailed_results = []
                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if paragraph and len(paragraph) > 50:  # Only include substantial paragraphs
                        detailed_results.append(paragraph)
                
                # If we have fewer paragraphs than requested results, try another search with different terms
                if len(detailed_results) < num_results:
                    additional_response = self.search.run(
                        f"{topic} comprehensive explanation details facts 2024 2025"
                    )
                    additional_paragraphs = additional_response.split('\n\n')
                    
                    for paragraph in additional_paragraphs:
                        paragraph = paragraph.strip()
                        if paragraph and len(paragraph) > 50 and paragraph not in detailed_results:
                            detailed_results.append(paragraph)
                            if len(detailed_results) >= num_results:
                                break
                
                detailed_results = sorted(detailed_results, key=len, reverse=True)[:num_results]
                search_results[topic] = detailed_results
                
            except Exception as e:
                search_results[topic] = [f"Search error: {str(e)}"]
        
        return search_results 