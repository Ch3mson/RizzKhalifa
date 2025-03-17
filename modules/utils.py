import os
import argparse
import traceback

def parse_arguments():
    """
    Parse command line arguments for the application.
    
    Returns:
        argparse.Namespace: The parsed command line arguments
    """
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
    parser.add_argument("--keep-temp", action="store_true", default=False,
                      help="Keep temporary files after execution (default: False)")
    
    return parser.parse_args()

def cleanup_temp_files():
    """
    Clean up the temp_files directory by removing all temporary files.
    Preserves the directory structure but deletes all contents.
    """
    try:
        temp_dir = os.path.join(os.getcwd(), "temp_files")
        if os.path.exists(temp_dir):
            
            file_count = 0
            for root, dirs, files in os.walk(temp_dir):
                file_count += len(files)
            
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")
            
            debug_dir = os.path.join(temp_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
    except Exception as e:
        print(f"Error during temp file cleanup: {e}")
        traceback.print_exc() 