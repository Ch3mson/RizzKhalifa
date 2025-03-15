#!/usr/bin/env python3

import sys
import argparse
import os
import traceback
from modules.assistant import ConversationAssistant

def parse_arguments():
    parser = argparse.ArgumentParser(description="Voice Assistant with Speaker Diarization")
    
    parser.add_argument("--diarization", action="store_true", default=True,
                      help="Enable speaker diarization (default: enabled)")
    parser.add_argument("--no-diarization", action="store_false", dest="diarization",
                      help="Disable speaker diarization")
    parser.add_argument("--speakers", type=int, default=2,
                      help="Expected number of speakers in conversations (default: 2)")
    parser.add_argument("--screen", action="store_true", default=False,
                      help="Enable screen capture for automatic facial recognition (default: disabled)")
    parser.add_argument("--debug", action="store_true", default=False,
                      help="Enable debug mode with verbose logging")
    parser.add_argument("--test", action="store_true", default=False,
                      help="Run in test mode to validate workflow functionality")
    parser.add_argument("--face-recheck", type=int, default=60,
                      help="Interval in seconds to check if the detected face is the same (default: 60)")
    
    return parser.parse_args()

def test_workflow():
    """Test the workflow functionality directly without audio input"""
    from modules.workflow import ConversationWorkflow
    print("\n===== TESTING WORKFLOW FUNCTIONALITY =====")
    
    # Create test directory
    log_dir = os.path.join(os.getcwd(), "test_results")
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize workflow
    workflow = ConversationWorkflow()
    
    # Test inputs that should trigger search
    test_inputs = [
        "Tell me about quantum computing",
        "What's happening with AI developments in 2024?",
        "I'm interested in learning about clean energy innovations"
    ]
    
    results = {}
    
    # Process each test input
    for i, test_input in enumerate(test_inputs):
        print(f"\n\n==== Test {i+1}: {test_input} ====")
        try:
            # Get baseline state
            if hasattr(workflow, 'graph') and workflow.graph:
                print("Using workflow graph to process")
                initial_state = workflow.state.copy()
                workflow.update_conversation(test_input)
                results[f"test_{i+1}"] = {
                    "input": test_input,
                    "success": True,
                    "has_knowledge_base": "knowledge_base" in workflow.state,
                    "kb_size": len(workflow.state.get("knowledge_base", {}))
                }
            else:
                print("⚠ Workflow graph not initialized correctly")
                results[f"test_{i+1}"] = {
                    "input": test_input,
                    "success": False,
                    "error": "Workflow graph not initialized"
                }
        except Exception as e:
            print(f"❌ Test {i+1} failed with error: {e}")
            traceback.print_exc()
            results[f"test_{i+1}"] = {
                "input": test_input,
                "success": False,
                "error": str(e)
            }
    
    # Print results summary
    print("\n===== TEST RESULTS SUMMARY =====")
    success_count = sum(1 for r in results.values() if r["success"])
    print(f"Successful tests: {success_count}/{len(test_inputs)}")
    
    for test_id, result in results.items():
        status = "✓" if result["success"] else "❌"
        kb_info = f", KB: {result['kb_size']} topics" if result["success"] and "kb_size" in result else ""
        error = f" - Error: {result['error']}" if not result["success"] and "error" in result else ""
        print(f"{status} {test_id}: '{result['input'][:30]}...'{kb_info}{error}")
    
    return success_count == len(test_inputs)

if __name__ == "__main__":
    try:
        args = parse_arguments()
        
        # Configure environment based on debug flag
        if args.debug:
            import logging
            logging.basicConfig(level=logging.DEBUG)
            print("Debug logging enabled")
            
            # Also enable more detailed exception reporting
            def excepthook(exc_type, exc_value, exc_traceback):
                print("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
            sys.excepthook = excepthook
        
        print(f"Starting Conversation Assistant...")
        
        # Handle test mode
        if args.test:
            success = test_workflow()
            sys.exit(0 if success else 1)
        
        # Normal execution
        if args.diarization:
            print(f"Speaker diarization enabled (expected speakers: {args.speakers})")
            print("You'll be asked to provide a 10-second voice sample for identification")
        else:
            print("Speaker diarization disabled")
            
        if args.screen:
            print("Screen capture enabled for facial recognition")
            print("Faces will be automatically detected and matched during conversation")
            print(f"Conversations will be saved in: {os.path.join(os.getcwd(), 'conversations')}/PERSON_ID/")
            print(f"Each conversation includes: conversation.txt, knowledge_base.txt, summary.txt, and more")
            print(f"Face will be rechecked every {args.face_recheck} seconds")
        
        # Initialize and run the assistant
        assistant = ConversationAssistant(
            use_diarization=args.diarization,
            expected_speakers=args.speakers,
            use_camera=args.screen
        )
        
        # If facial recognition is enabled, set the recheck interval
        if args.screen and hasattr(assistant, 'facial_recognition') and assistant.facial_recognition:
            assistant.facial_recognition.set_recheck_interval(args.face_recheck)
            
        assistant.run()
        
    except KeyboardInterrupt:
        print("\nExiting program due to keyboard interrupt")
    except Exception as e:
        print(f"Startup error: {e}")
        if 'args' in locals() and args.debug:
            traceback.print_exc() 