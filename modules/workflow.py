#!/usr/bin/env python3

from typing import List, Dict
import concurrent.futures
import traceback
import os
import time

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
      1) Process input to determine if it's worth processing
      2) If worth processing, summarize conversation
      3) Extract topics and personal info
      4) If search is needed, search for relevant topics
      5) Present results (save them)
    """
    def __init__(self):
        # Initialize agents
        self.processor_agent = ProcessorAgent()
        self.summarizer_agent = SummarizerAgent()
        self.topic_extraction_agent = TopicExtractionAgent()
        self.personal_info_agent = PersonalInfoAgent()  
        self.response_generation_agent = ResponseGenerationAgent()
        self.search_agent = SearchAgent()
        
        # Initialize output helper
        self.text_output = TextOutput()
        
        # Initialize state
        self.state = CONVERSATION_STATE_SCHEMA.copy()
        
        # Track speaker information
        self.speaker_segments = []

        # Set up the graph
        self.setup_graph()

    def update_speaker_segments(self, segments: List[Dict]):
        """
        Update the speaker segments in the workflow state
        
        Args:
            segments: List of diarized segments with speaker information
        """
        try:
            self.speaker_segments = segments
            
            # Update the state with speaker information
            if "speaker_segments" not in self.state:
                self.state["speaker_segments"] = []
                
            self.state["speaker_segments"].extend(segments)
            
            # Log the speakers
            speakers = set()
            persons = set()
            
            for segment in segments:
                if "speaker" in segment:
                    speakers.add(segment["speaker"])
                if "person" in segment:
                    persons.add(segment["person"])
                    
            if speakers:
                print(f"Updated workflow with speakers: {', '.join(speakers)}")
            if persons:
                print(f"Identified persons: {', '.join(persons)}")
                
        except Exception as e:
            print(f"Error updating speaker segments: {e}")

    def setup_graph(self):
        """
        Create a linear (serial) workflow:
            summarize_conversation
                -> extract_topics
                -> search_topics
                -> extract_personal_info
                -> present_results
                -> END
        This avoids concurrency, so we won't get the 
        "Can receive only one value per step" error.
        """
        try:
            print("Setting up LangGraph workflow...")
            # Add a version check for langgraph
            try:
                import langgraph
                print(f"Using LangGraph version: {langgraph.__version__}")
            except (ImportError, AttributeError):
                print("Could not determine LangGraph version")
            
            # Use typed dict to ensure proper schema
            workflow = StateGraph(ConversationState)

            print("Adding nodes to graph...")
            # Wrap all node functions in try-except blocks for better error handling
            def safe_summarize(state):
                try:
                    return self.summarize_conversation(state)
                except Exception as e:
                    print(f"Error in summarize node: {e}")
                    traceback.print_exc()
                    # Return unmodified state to continue the workflow
                    return state
                    
            def safe_extract_topics(state):
                try:
                    return self.extract_topics(state)
                except Exception as e:
                    print(f"Error in extract_topics node: {e}")
                    traceback.print_exc()
                    # Set empty topics and continue
                    state["topics"] = []
                    return state
                    
            def safe_search(state):
                try:
                    return self.search_for_topics(state)
                except Exception as e:
                    print(f"Error in search node: {e}")
                    traceback.print_exc()
                    # Set empty knowledge base and continue
                    state["knowledge_base"] = {}
                    return state
                    
            def safe_extract_personal_info(state):
                try:
                    return self.extract_personal_info(state)
                except Exception as e:
                    print(f"Error in personal_info node: {e}")
                    traceback.print_exc()
                    # Set empty personal_info and continue
                    state["personal_info"] = []
                    return state
                    
            def safe_present(state):
                try:
                    return self.present_results(state)
                except Exception as e:
                    print(f"Error in present_results node: {e}")
                    traceback.print_exc()
                    return state
            
            # Add wrapped nodes for better error handling
            workflow.add_node("summarize_conversation", safe_summarize)
            workflow.add_node("extract_topics", safe_extract_topics)
            workflow.add_node("search_topics", safe_search)
            workflow.add_node("extract_personal_info", safe_extract_personal_info)
            workflow.add_node("present_results", safe_present)

            # Create edges
            print("Creating graph edges...")
            workflow.add_edge("summarize_conversation", "extract_topics")
            workflow.add_edge("extract_topics", "search_topics")
            workflow.add_edge("search_topics", "extract_personal_info")
            workflow.add_edge("extract_personal_info", "present_results")
            workflow.add_edge("present_results", END)

            print("Setting entry point...")
            workflow.set_entry_point("summarize_conversation")
            
            print("Compiling LangGraph...")
            self.graph = workflow.compile()
            
            if self.graph:
                print("✓ LangGraph workflow successfully initialized")
            else:
                print("! Failed to compile LangGraph")
        except Exception as e:
            print(f"! Graph setup error: {e}")
            traceback.print_exc()
            self.graph = None
            print("Will fall back to linear execution")
    def summarize_conversation(self, state: ConversationState) -> ConversationState:
        try:
            # Pass speaker segments to the summarizer if available
            if "speaker_segments" in state and state["speaker_segments"]:
                summary = self.summarizer_agent.summarize(
                    state["conversation"], 
                    speaker_segments=state["speaker_segments"]
                )
            else:
                summary = self.summarizer_agent.summarize(state["conversation"])
                
            state["summary"] = summary
        except Exception as e:
            print(f"Summarizing error: {e}")
            state["summary"] = ""
        return state
    
    def extract_topics(self, state: ConversationState) -> ConversationState:
        try:
            topics = self.topic_extraction_agent.extract_topics(state["summary"])
            state["topics"] = topics
        except Exception as e:
            print(f"Topic extraction error: {e}")
            state["topics"] = []
        return state

    def extract_personal_info(self, state: ConversationState) -> ConversationState:
        try:
            personal_info = self.personal_info_agent.extract_personal_info(state["summary"])
            state["personal_info"] = personal_info
        except Exception as e:
            print(f"Personal info error: {e}")
        return state

    def search_for_topics(self, state: ConversationState) -> ConversationState:
        """
        Search for topics to generate knowledge base.
        This function only searches for topics that were explicitly stated as preferences,
        hobbies, or professional/educational activities.
        """
        try:
            # Get the last processed input
            last_processed = state.get("last_processed", "")
            if not last_processed:
                print("No recent input to analyze for search need")
                return state
                
            print(f"Analyzing for explicit preferences or interests: '{last_processed[:100]}...'")
            
            # Check if input requires search using processor agent
            processor_result = None
            try:
                processor_result = self.processor_agent.process(last_processed)
                print(f"Processor analysis result: {processor_result}")
            except Exception as e:
                print(f"Error in processor analysis: {e}")
                import traceback
                traceback.print_exc()
            
            # Determine if we need to search based on STRICT criteria
            should_search = False
            custom_topics = []
            
            if processor_result and processor_result.get("category") == "SEARCH_TOPIC":
                should_search = True
                custom_topics = processor_result.get("topics", [])
                print(f"Explicit preference detected, will search for: {custom_topics}")
            else:
                # No direct search from this input - but don't look at conversation context
                # We're only searching for explicitly stated preferences
                print("No explicit preference detected in this input, skipping search")
                
            # Skip search if no need identified
            if not should_search or not custom_topics:
                print("No search needed - no explicit preferences identified")
                return state
            
            # Save the identified topics to state before searching
            # This ensures topics are saved even if search fails
            if custom_topics:
                # Format topics as proper objects if they're not already
                formatted_topics = []
                for topic in custom_topics:
                    if isinstance(topic, str):
                        formatted_topics.append({
                            "name": topic,
                            "aspect": "explicit interest or preference"
                        })
                    else:
                        formatted_topics.append(topic)
                
                # Append to existing topics or create new list
                if "topics" not in state or not state["topics"]:
                    state["topics"] = formatted_topics
                else:
                    # Check for duplicates before adding
                    existing_topic_names = [t.get("name", "") for t in state["topics"]]
                    for topic in formatted_topics:
                        if topic.get("name") not in existing_topic_names:
                            state["topics"].append(topic)
                
                # Save topics to files immediately
                self._save_topics_to_file()
            
            # Perform search with timeout protection
            print(f"Executing search for explicit preferences: {custom_topics}")
            try:
                import concurrent.futures
                import time
                
                # Create a clean list of topics (strip extra spaces, etc.)
                search_topics = [topic.strip() if isinstance(topic, str) else topic.get("name", "").strip() 
                                for topic in custom_topics if topic]
                search_topics = [t for t in search_topics if t]  # Remove any empty strings
                
                if not search_topics:
                    print("No valid search topics after cleaning")
                    return state
                
                # Set a timeout for the search operation (30 seconds)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Submit the search task
                    future = executor.submit(self.search_agent.search_topics, search_topics)
                    
                    # Wait for results with timeout
                    try:
                        search_start = time.time()
                        kb = future.result(timeout=30)
                        search_duration = time.time() - search_start
                        
                        # Update state with knowledge base
                        if kb:
                            state["knowledge_base"] = kb
                            print(f"✓ Search completed in {search_duration:.2f}s, found information for {len(kb)} topics:")
                            for topic, snippets in kb.items():
                                print(f"  - {topic}: {len(snippets)} snippets")
                            
                            # Save knowledge base to files immediately
                            self._save_knowledge_base_to_file()
                        else:
                            print("Search returned empty knowledge base")
                            
                    except concurrent.futures.TimeoutError:
                        print("⚠ Search operation timed out after 30 seconds")
                        # Keep any existing knowledge base
                
            except Exception as e:
                print(f"Error during search execution: {e}")
                import traceback
                traceback.print_exc()
                # Keep existing knowledge base if search fails
                
            # Return the updated state
            return state
                
        except Exception as e:
            print(f"Error in search_for_topics: {e}")
            import traceback
            traceback.print_exc()
            return state

    def present_results(self, state: ConversationState) -> ConversationState:
        self.text_output.save_to_file(state)
        return state

    ###########################################################################
    #                           EXTERNAL METHODS
    ###########################################################################

    def update_conversation(self, new_text: str):
        """
        Append new_text to the conversation in state, then run the graph.
        Also saves the data to text files immediately to prevent data loss on interruption.
        """
        if not new_text or not new_text.strip():
            print("Empty input, skipping processing")
            return
            
        # Check if the input is meaningful or just casual conversation
        try:
            process_result = self.processor_agent.process(new_text)
            category = process_result.get("category", "SKIP")
            explanation = process_result.get("explanation", "")
            
            if category == "SKIP":
                print(f"Skipping processing: {explanation}")
                # Still add to conversation but don't run the workflow
                if self.state["conversation"]:
                    self.state["conversation"] += "\n" + new_text
                else:
                    self.state["conversation"] = new_text
                return
                
            print(f"Processing input as '{category}': {explanation}")
        except Exception as e:
            print(f"Error checking process category: {e}. Will attempt to process anyway.")
            import traceback
            traceback.print_exc()
            
        # Add to the conversation buffer
        if self.state["conversation"]:
            self.state["conversation"] += "\n" + new_text
        else:
            self.state["conversation"] = new_text

        # Track what we're processing
        self.state["last_processed"] = new_text
        
        # Save conversation to text file immediately - before running the workflow
        # This ensures data is preserved even if there's an interruption
        self._save_conversation_to_file()
        
        # Run the entire workflow
        try:
            if self.graph:
                print("Running input through LangGraph workflow...")
                result = self.graph.invoke(self.state)
                if result:
                    self.state = result
                    
                    # After workflow execution, save any updated data to files immediately
                    if "knowledge_base" in self.state and self.state["knowledge_base"]:
                        kb_size = len(self.state["knowledge_base"])
                        print(f"Knowledge base updated with {kb_size} topics")
                        # Save knowledge base to text file
                        self._save_knowledge_base_to_file()
                    
                    # Save topics if available
                    if "topics" in self.state and self.state["topics"]:
                        # Save topics to text file
                        self._save_topics_to_file()
                else:
                    print("WARNING: LangGraph returned None, state not updated")
            else:
                # If the graph isn't compiled, fallback to the old linear approach
                print("Graph not available; fallback to linear pipeline.")
                self._run_linear_workflow()
                
                # After linear workflow, save any updated data
                self._save_knowledge_base_to_file()
                self._save_topics_to_file()
                
        except Exception as e:
            print(f"Workflow error: {e}")
            import traceback
            traceback.print_exc()
            # Try linear approach as fallback if graph fails
            try:
                print("Trying linear approach as fallback...")
                self._run_linear_workflow()
                
                # After fallback execution, save any updated data
                self._save_knowledge_base_to_file()
                self._save_topics_to_file()
                
            except Exception as e2:
                print(f"Linear fallback also failed: {e2}")
                traceback.print_exc()
                
    def _save_conversation_to_file(self):
        """Save the current conversation to text files for all recognized persons"""
        try:
            # Get conversation and speaker segments
            conversation = self.state.get("conversation", "")
            speaker_segments = self.state.get("speaker_segments", [])
            
            # Get unique persons from the speaker segments
            unique_persons = set()
            for segment in speaker_segments:
                if 'person' in segment and segment['person']:
                    unique_persons.add(segment['person'])
            
            # If no persons identified, use default Person with timestamp
            if not unique_persons:
                # Check if we previously saved under a generic name
                if hasattr(self, '_default_person_id'):
                    unique_persons = {self._default_person_id}
                else:
                    # Create a stable ID once per session
                    current_time = int(time.time())
                    self._default_person_id = f"Person_{current_time}"
                    unique_persons = {self._default_person_id}
            
            # Base directory for conversations
            base_dir = os.path.join(os.getcwd(), "conversations")
            os.makedirs(base_dir, exist_ok=True)
            
            # Save for each unique person
            for person in unique_persons:
                # Clean up person name for consistency
                if hasattr(self, '_person_id_map') and person in self._person_id_map:
                    person = self._person_id_map[person]
                elif person.startswith("Person_") or person.startswith("Unknown"):
                    parts = person.split("_")
                    if len(parts) > 1:
                        base_name = f"{parts[0]}_{parts[1]}"
                        if not hasattr(self, '_person_id_map'):
                            self._person_id_map = {}
                        self._person_id_map[person] = base_name
                        person = base_name
                
                # Create person directory
                person_dir = os.path.join(base_dir, person)
                os.makedirs(person_dir, exist_ok=True)
                
                # Get the next conversation number
                conversation_files = [f for f in os.listdir(person_dir) 
                                   if f.startswith("conversation_") and f.endswith(".txt")]
                next_number = 1
                if conversation_files:
                    numbers = [int(f.split("_")[1].split(".")[0]) for f in conversation_files]
                    next_number = max(numbers) + 1
                
                # Save new conversation with incremented number
                conversation_path = os.path.join(person_dir, f"conversation_{next_number}.txt")
                
                with open(conversation_path, 'w') as f:
                    f.write(f"CONVERSATION {next_number} WITH {person.upper()}\n")
                    f.write("="*80 + "\n\n")
                    
                    # Add the structured speaker segments if available
                    if speaker_segments:
                        for segment in speaker_segments:
                            speaker = segment.get('speaker', 'Unknown')
                            segment_person = segment.get('person', speaker)
                            text = segment.get('text', '')
                            f.write(f"[{segment_person}]: {text}\n")
                    else:
                        # Otherwise add the raw conversation text
                        f.write(conversation)
                        
                print(f"✓ Saved conversation {next_number} for {person}")
                
        except Exception as e:
            print(f"Error saving conversation to file: {e}")
            traceback.print_exc()
            
    def _save_knowledge_base_to_file(self):
        """Save the current knowledge base to text files for all recognized persons"""
        try:
            # Get knowledge base and speaker segments
            knowledge_base = self.state.get("knowledge_base", {})
            speaker_segments = self.state.get("speaker_segments", [])
            
            # Skip if no knowledge base
            if not knowledge_base:
                return
                
            # Get unique persons from the speaker segments
            unique_persons = set()
            for segment in speaker_segments:
                if 'person' in segment and segment['person']:
                    unique_persons.add(segment['person'])
            
            # If no persons identified, use default Person with timestamp
            if not unique_persons:
                if hasattr(self, '_default_person_id'):
                    unique_persons = {self._default_person_id}
                else:
                    # Create a stable ID once per session
                    current_time = int(time.time())
                    self._default_person_id = f"Person_{current_time}"
                    unique_persons = {self._default_person_id}
            
            # Base directory for conversations
            base_dir = os.path.join(os.getcwd(), "conversations")
            os.makedirs(base_dir, exist_ok=True)
            
            # Save for each unique person
            for person in unique_persons:
                # Clean up person name for consistency, removing any extra timestamps
                # Use the map if available
                if hasattr(self, '_person_id_map') and person in self._person_id_map:
                    person = self._person_id_map[person]
                elif person.startswith("Person_") or person.startswith("Unknown"):
                    # Extract just the base identifier without multiple timestamps
                    parts = person.split("_")
                    if len(parts) > 1:
                        # Keep just the first segment (Person or Unknown) and the next segment
                        base_name = f"{parts[0]}_{parts[1]}"
                        
                        # Store this cleaned name for future use
                        if not hasattr(self, '_person_id_map'):
                            self._person_id_map = {}
                        self._person_id_map[person] = base_name
                        person = base_name
                
                # Create person directory
                person_dir = os.path.join(base_dir, person)
                os.makedirs(person_dir, exist_ok=True)
                
                # Load existing knowledge if any
                knowledge_path = os.path.join(person_dir, "knowledge_base.txt")
                existing_knowledge = set()
                if os.path.exists(knowledge_path):
                    with open(knowledge_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('KNOWLEDGE BASE FOR') and not line.startswith('='):
                                existing_knowledge.add(line)
                
                # Add new knowledge
                if knowledge_base:
                    # Convert knowledge items to strings and add to set
                    new_knowledge = set()
                    for topic, snippets in knowledge_base.items():
                        for s in snippets:
                            new_knowledge.add(f"{topic}: {s}")
                    
                    # Combine existing and new knowledge
                    all_knowledge = existing_knowledge.union(new_knowledge)
                    
                    # Write combined knowledge
                    with open(knowledge_path, 'w') as f:
                        f.write(f"KNOWLEDGE BASE FOR {person.upper()}\n")
                        f.write("="*80 + "\n\n")
                        for item in sorted(all_knowledge):
                            f.write(f"{item}\n")
                    
                    print(f"✓ Extended knowledge base for {person}")
                
        except Exception as e:
            print(f"Error saving knowledge base to file: {e}")
            traceback.print_exc()
            
    def _save_topics_to_file(self):
        """Save the extracted topics to text files for all recognized persons"""
        try:
            # Get topics and speaker segments
            topics = self.state.get("topics", [])
            speaker_segments = self.state.get("speaker_segments", [])
            
            # Get unique persons from the speaker segments
            unique_persons = set()
            for segment in speaker_segments:
                if 'person' in segment and segment['person']:
                    unique_persons.add(segment['person'])
            
            # If no persons identified, use default Person with timestamp
            if not unique_persons:
                if hasattr(self, '_default_person_id'):
                    unique_persons = {self._default_person_id}
                else:
                    current_time = int(time.time())
                    self._default_person_id = f"Person_{current_time}"
                    unique_persons = {self._default_person_id}
            
            # Base directory for conversations
            base_dir = os.path.join(os.getcwd(), "conversations")
            os.makedirs(base_dir, exist_ok=True)
            
            # Save for each unique person
            for person in unique_persons:
                # Clean up person name for consistency
                if hasattr(self, '_person_id_map') and person in self._person_id_map:
                    person = self._person_id_map[person]
                elif person.startswith("Person_") or person.startswith("Unknown"):
                    parts = person.split("_")
                    if len(parts) > 1:
                        base_name = f"{parts[0]}_{parts[1]}"
                        if not hasattr(self, '_person_id_map'):
                            self._person_id_map = {}
                        self._person_id_map[person] = base_name
                        person = base_name
                
                # Create person directory
                person_dir = os.path.join(base_dir, person)
                os.makedirs(person_dir, exist_ok=True)
                
                # Load existing topics if any
                topics_path = os.path.join(person_dir, "topics.txt")
                existing_topics = {
                    'Likes': set(),
                    'Dislikes': set(),
                    'School': set(),
                    'Hobbies': set(),
                    'Professional': set(),
                    'Skills': set()
                }
                
                if os.path.exists(topics_path):
                    current_category = None
                    with open(topics_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                if line.endswith(':'):  # Category header
                                    current_category = line[:-1]
                                elif current_category and current_category in existing_topics:
                                    # Skip bullet points and clean up the line
                                    topic = line.lstrip('- ').strip()
                                    if topic:
                                        existing_topics[current_category].add(topic)
                
                # Add new topics
                if topics:
                    # Convert topics to categorized format
                    categorized_topics = {
                        'Likes': set(),
                        'Dislikes': set(),
                        'School': set(),
                        'Hobbies': set(),
                        'Professional': set(),
                        'Skills': set()
                    }
                    
                    for topic in topics:
                        if isinstance(topic, dict):
                            category = topic.get('category', '')
                            name = topic.get('name', '')
                            description = topic.get('description', '')
                            
                            if category in categorized_topics and name:
                                topic_text = name
                                if description:
                                    topic_text += f" - {description}"
                                categorized_topics[category].add(topic_text)
                    
                    # Combine existing and new topics
                    for category in categorized_topics:
                        categorized_topics[category] = existing_topics[category].union(categorized_topics[category])
                    
                    # Write combined topics
                    with open(topics_path, 'w') as f:
                        f.write(f"TOPICS FOR {person.upper()}\n")
                        f.write("="*80 + "\n\n")
                        
                        # Write each category that has topics
                        for category, items in categorized_topics.items():
                            if items:
                                f.write(f"{category}:\n")
                                for item in sorted(items):
                                    f.write(f"- {item}\n")
                                f.write("\n")
                    
                    print(f"✓ Extended topics for {person}")
                
        except Exception as e:
            print(f"Error saving topics to file: {e}")
            traceback.print_exc()

    def _run_linear_workflow(self):
        """Run workflow in a simple linear fashion without the graph"""
        print("Running linear workflow...")
        self.state = self.summarize_conversation(self.state)
        print("✓ Summarized conversation")
        self.state = self.extract_topics(self.state)
        print("✓ Extracted topics")
        self.state = self.search_for_topics(self.state)
        print("✓ Searched for topics")
        self.state = self.extract_personal_info(self.state)
        print("✓ Extracted personal info")
        self.state = self.present_results(self.state)
        print("✓ Results saved")

    def generate_knowledge_response(self, recent_snippet: str) -> str:
        """
        Take the most recent snippet of conversation, plus the knowledge base, 
        to generate a short assistant answer.
        """
        try:
            # Flatten knowledge_base for the prompt
            kb_text = ""
            for topic, snippets in self.state["knowledge_base"].items():
                kb_text += f"{topic}:\n"
                for s in snippets:
                    kb_text += f"- {s}\n"
            
            return self.response_generation_agent.generate_response(
                recent_snippet, kb_text
            )
        except Exception as e:
            print(f"Response generation error: {e}")
            return "I'm having trouble generating a response right now." 