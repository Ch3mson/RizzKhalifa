#!/usr/bin/env python3

from typing import List, Dict, Any
import concurrent.futures
import time
import os
import traceback  

from langgraph.graph import StateGraph, END

from modules.config import ConversationState, CONVERSATION_STATE_SCHEMA
from modules.output import TextOutput
from modules.agents import (
    SearchAgent,
    SummarizerAgent,
    TopicExtractionAgent,
    PersonalInfoAgent,
    ResponseGenerationAgent,
    ProcessorAgent
)


class ConversationWorkflow:
    """
    Builds and executes a graph-based workflow:
      1) check_category (router)
          -> if category == "SKIP": restart sequence
          -> else: normal flow:
                summarize_conversation
                -> extract_topics
                -> search_topics
                -> extract_personal_info
                -> present_results
                -> END
    """
    def __init__(self):
        self.processor_agent = ProcessorAgent()
        self.summarizer_agent = SummarizerAgent()
        self.topic_extraction_agent = TopicExtractionAgent()
        self.personal_info_agent = PersonalInfoAgent()
        self.response_generation_agent = ResponseGenerationAgent()
        self.search_agent = SearchAgent()

        self.text_output = TextOutput()
        self.state = CONVERSATION_STATE_SCHEMA.copy()
        self.speaker_segments: List[Dict[str, Any]] = []
        self.setup_graph()

    def update_speaker_segments(self, segments: List[Dict]):
        """Update the speaker segments in the workflow state."""
        try:
            self.speaker_segments = segments
            if "speaker_segments" not in self.state:
                self.state["speaker_segments"] = []
            self.state["speaker_segments"].extend(segments)

            speakers = {seg["speaker"] for seg in segments if "speaker" in seg}
            persons = {seg["person"] for seg in segments if "person" in seg}
            if speakers:
                print(f"Updated workflow with speakers: {', '.join(speakers)}")
            if persons:
                print(f"Identified persons: {', '.join(persons)}")

        except Exception as e:
            print(f"Error updating speaker segments: {e}")
            traceback.print_exc()

    def setup_graph(self):
        """
        Create a workflow with conditional routing:
            check_category
              -> if category == "SKIP": restart (loop back)
              -> else: continue normal processing
        """
        try:
            try:
                import langgraph
                print(f"Using LangGraph version: {langgraph.__version__}")
            except (ImportError, AttributeError):
                print("Could not determine LangGraph version")

            workflow = StateGraph(ConversationState)

            def check_category(state: ConversationState) -> ConversationState:
                """
                Decide how to route based on 'category' in the state.
                If category == 'SKIP', we attempt to restart.
                Otherwise, we continue the normal flow.
                """
                state_copy = state.copy()
                category = state_copy.get("category", "")

                if category == "SKIP":
                    state_copy.pop("category", None)
                    restart_count = state_copy.get("restart_count", 0) + 1
                    state_copy["restart_count"] = restart_count

                    if restart_count > 3:
                        state_copy["restart_count"] = 0
                        state_copy["_routing"] = "continue"
                        return state_copy

                    state_copy["_routing"] = "restart"
                    return state_copy
                else:
                    state_copy.pop("category", None)  
                    state_copy["restart_count"] = 0
                    state_copy["_routing"] = "continue"
                    return state_copy

            def route_fn(state: ConversationState):
                return state.get("_routing", "continue")

            def safe_summarize(state: ConversationState) -> ConversationState:
                try:
                    return self.summarize_conversation(state)
                except Exception as e:
                    print(f"Error in summarize_conversation node: {e}")
                    traceback.print_exc()
                    copy_ = state.copy()
                    copy_["summary"] = ""
                    return copy_

            def safe_extract_topics(state: ConversationState) -> ConversationState:
                try:
                    return self.extract_topics(state)
                except Exception as e:
                    print(f"Error in extract_topics node: {e}")
                    traceback.print_exc()
                    copy_ = state.copy()
                    copy_["topics"] = []
                    return copy_

            def safe_search_topics(state: ConversationState) -> ConversationState:
                try:
                    return self.search_for_topics(state)
                except Exception as e:
                    print(f"Error in search_for_topics node: {e}")
                    traceback.print_exc()
                    copy_ = state.copy()
                    copy_["knowledge_base"] = {}
                    return copy_

            def safe_extract_personal_info(state: ConversationState) -> ConversationState:
                try:
                    return self.extract_personal_info(state)
                except Exception as e:
                    print(f"Error in extract_personal_info node: {e}")
                    traceback.print_exc()
                    copy_ = state.copy()
                    copy_["personal_info"] = {}
                    return copy_

            def safe_present(state: ConversationState) -> ConversationState:
                try:
                    return self.present_results(state)
                except Exception as e:
                    print(f"Error in present_results node: {e}")
                    traceback.print_exc()
                    return state.copy()

            workflow.add_node("check_category", check_category)
            workflow.add_node("summarize_conversation", safe_summarize)
            workflow.add_node("extract_topics", safe_extract_topics)
            workflow.add_node("search_topics", safe_search_topics)
            workflow.add_node("extract_personal_info", safe_extract_personal_info)
            workflow.add_node("present_results", safe_present)

            workflow.add_conditional_edges(
                "check_category",
                route_fn,
                {
                    "restart": "check_category",
                    "continue": "summarize_conversation"
                }
            )

            workflow.add_edge("summarize_conversation", "extract_topics")
            workflow.add_edge("extract_topics", "search_topics")
            workflow.add_edge("search_topics", "extract_personal_info")
            workflow.add_edge("extract_personal_info", "present_results")
            workflow.add_edge("present_results", END)

            workflow.set_entry_point("check_category")
            self.graph = workflow.compile()
            
        except Exception as e:
            print(f"! Graph setup error: {e}")
            traceback.print_exc()
            self.graph = None

    def summarize_conversation(self, state: ConversationState) -> ConversationState:
        """Summarize the conversation text in state['conversation']."""
        try:
            copy_ = state.copy()
            conversation = copy_.get("conversation", "")
            if not conversation.strip():
                copy_["summary"] = ""
                return copy_
            if "speaker_segments" in copy_ and copy_["speaker_segments"]:
                summary = self.summarizer_agent.summarize(
                    conversation, speaker_segments=copy_["speaker_segments"]
                )
            else:
                summary = self.summarizer_agent.summarize(conversation)
            copy_["summary"] = summary
            return copy_
        except Exception as e:
            traceback.print_exc()
            c2 = state.copy()
            c2["summary"] = ""
            return c2

    def extract_topics(self, state: ConversationState) -> ConversationState:
        """Extract topics from state['summary'] using the TopicExtractionAgent."""
        try:
            copy_ = state.copy()
            summary = copy_.get("summary", "")
            if not summary.strip():
                copy_["topics"] = []
                return copy_
            topics = self.topic_extraction_agent.extract_topics(summary)
            copy_["topics"] = topics
            return copy_
        except Exception as e:
            traceback.print_exc()
            c2 = state.copy()
            c2["topics"] = []
            return c2

    def search_for_topics(self, state: ConversationState) -> ConversationState:
        """
        Looks at state['last_processed'] plus any topics, decides if we need to do a search
        by analyzing user statements. If needed, populates knowledge_base.
        """
        copy_ = state.copy()
        last_processed = copy_.get("last_processed", "")
        if not last_processed.strip():
            print("No last_processed text to analyze for search.")
            return copy_

        try:
            process_res = self.processor_agent.process(last_processed)
            if process_res.get("category") == "SEARCH_TOPIC":
                new_topics = process_res.get("topics", [])
                if new_topics:
                    existing = copy_.get("topics", [])
                    existing_names = {t.get("name") for t in existing if isinstance(t, dict)}
                    for t in new_topics:
                        if isinstance(t, dict) and t.get("name") not in existing_names:
                            existing.append(t)
                    copy_["topics"] = existing

                    to_search = []
                    for t in new_topics:
                        if isinstance(t, dict):
                            nm = t.get("name", "").strip()
                            if nm:
                                to_search.append(nm)

                    if to_search:
                        print(f"Searching for explicit user topics: {to_search}")
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(self.search_agent.search_topics, to_search)
                            try:
                                kb_result = future.result(timeout=30)
                                if kb_result:
                                    old_kb = copy_.get("knowledge_base", {})
                                    for ktopic, snips in kb_result.items():
                                        if ktopic not in old_kb:
                                            old_kb[ktopic] = snips
                                        else:
                                            old_kb[ktopic].extend(snips)
                                    copy_["knowledge_base"] = old_kb
                            except concurrent.futures.TimeoutError:
                                print("Search agent took too long (>30s). Skipping search.")
            else:
                print("No explicit search topic found in user statement.")
        except Exception as e:
            print(f"Error in search_for_topics: {e}")
            traceback.print_exc()

        return copy_

    def extract_personal_info(self, state: ConversationState) -> ConversationState:
        """Extract personal information from state['summary']."""
        copy_ = state.copy()
        summary = copy_.get("summary", "")
        if not summary.strip():
            print("No summary to extract personal info from.")
            copy_["personal_info"] = []
            return copy_

        try:
            pinfo = self.personal_info_agent.extract_personal_info(summary)
            copy_["personal_info"] = pinfo
        except Exception as e:
            print(f"Error extracting personal info: {e}")
            traceback.print_exc()
            copy_["personal_info"] = []
        return copy_

    def present_results(self, state: ConversationState) -> ConversationState:
        copy_ = state.copy()
        try:
            self.text_output.save_to_file(copy_, "workflow_output.txt")
        except Exception as e:
            print(f"Error in present_results: {e}")
            traceback.print_exc()
        return copy_


    def update_conversation(self, new_text: str):
        """
        Append new_text to state['conversation'], set last_processed,
        then run the entire graph.
        """
        if not new_text.strip():
            return

        if self.state["conversation"]:
            self.state["conversation"] += "\n" + new_text
        else:
            self.state["conversation"] = new_text

        self.state["last_processed"] = new_text

        if self.graph:
            try:
                result = self.graph.invoke(self.state)
                if result:
                    self.state = result
                else:
                    print("Graph returned None; no updates applied.")
            except Exception as e:
                traceback.print_exc()
                self._run_linear_fallback()
        else:
            self._run_linear_fallback()

    def _run_linear_fallback(self):
        """If graph fails, run each node in sequence (serially)."""
        st = self.summarize_conversation(self.state)
        st = self.extract_topics(st)
        st = self.search_for_topics(st)
        st = self.extract_personal_info(st)
        st = self.present_results(st)
        self.state = st

    def generate_knowledge_response(self, recent_snippet: str) -> str:
        """
        Produce a short assistant answer referencing state['knowledge_base'].
        """
        try:
            kb_text = ""
            kb = self.state.get("knowledge_base", {})
            for topic, snippets in kb.items():
                kb_text += f"{topic}:\n"
                for snippet in snippets:
                    kb_text += f"- {snippet}\n"
            return self.response_generation_agent.generate_response(recent_snippet, kb_text)
        except Exception as e:
            print(f"Response generation error: {e}")
            traceback.print_exc()
            return "I'm having trouble generating a response right now."
