#!/usr/bin/env python3
"""
FastAPI server to run the Cursor Assistant remotely.
This allows the cursor_main.py script to be triggered via HTTP endpoints.
"""

import os
import sys
import time
import threading
import subprocess
import uvicorn
from typing import Dict, Optional, List
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import psutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cursor_main import CursorAssistant, parse_arguments

app = FastAPI(
    title="Cursor Assistant API",
    description="API to remotely control the Cursor Dating Assistant",
    version="1.0.0"
)

assistant_process = None
assistant_thread = None
assistant_instance = None
is_running = False
process_output = []
max_output_lines = 1000 


class AssistantConfig(BaseModel):
    """Configuration options for starting the assistant"""
    diarization: bool = True
    speakers: int = 2
    screen: bool = False
    debug: bool = False


class AssistantStatus(BaseModel):
    """Status response for the assistant"""
    running: bool
    pid: Optional[int] = None
    uptime: Optional[float] = None
    start_time: Optional[str] = None
    memory_usage: Optional[str] = None
    cpu_percent: Optional[float] = None


def run_assistant_in_thread(config: AssistantConfig):
    """Run the assistant in a background thread"""
    global assistant_instance, is_running, process_output
    
    try:
        process_output = []
        
        class OutputCapture:
            def write(self, text):
                global process_output
                if text.strip(): 
                    process_output.append(text)
                    if len(process_output) > max_output_lines:
                        process_output = process_output[-max_output_lines:]
                sys.__stdout__.write(text)
                sys.__stdout__.flush()
            
            def flush(self):
                sys.__stdout__.flush()
        
        sys.stdout = OutputCapture()
        
        assistant_instance = CursorAssistant(
            use_diarization=config.diarization,
            expected_speakers=config.speakers,
            use_camera=config.screen
        )
        
        is_running = True
        
        assistant_instance.run()
        
    except Exception as e:
        import traceback
        error_msg = f"Error running assistant: {str(e)}\n{traceback.format_exc()}"
        process_output.append(error_msg)
        print(error_msg)
    finally:
        is_running = False
        sys.stdout = sys.__stdout__


def run_assistant_subprocess(config: AssistantConfig):
    """Run the assistant as a separate subprocess"""
    global assistant_process, is_running, process_output
    
    cmd = [sys.executable, "cursor_main.py"]
    
    if not config.diarization:
        cmd.append("--no-diarization")
    
    if config.speakers != 2:
        cmd.extend(["--speakers", str(config.speakers)])
    
    if config.screen:
        cmd.append("--screen")
    
    if config.debug:
        cmd.append("--debug")
    
    try:
        process_output = []
        
        assistant_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1, 
            universal_newlines=True
        )
        
        is_running = True
        
        for line in assistant_process.stdout:
            process_output.append(line.strip())
            if len(process_output) > max_output_lines:
                process_output = process_output[-max_output_lines:]
            print(line, end="") 
        
        assistant_process.wait()
        return_code = assistant_process.returncode
        
        if return_code != 0:
            process_output.append(f"Process exited with code {return_code}")
            
    except Exception as e:
        import traceback
        error_msg = f"Error running assistant subprocess: {str(e)}\n{traceback.format_exc()}"
        process_output.append(error_msg)
        print(error_msg)
    finally:
        is_running = False
        assistant_process = None


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with basic API information"""
    return {
        "name": "Cursor Assistant API",
        "version": "1.0.0",
        "description": "Control the Cursor Dating Assistant remotely",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "This information"},
            {"path": "/start", "method": "POST", "description": "Start the assistant"},
            {"path": "/stop", "method": "POST", "description": "Stop the assistant"},
            {"path": "/status", "method": "GET", "description": "Get assistant status"},
            {"path": "/logs", "method": "GET", "description": "Get assistant logs"}
        ]
    }


@app.post("/start", response_class=JSONResponse)
async def start_assistant(config: AssistantConfig, background_tasks: BackgroundTasks):
    """Start the Cursor Assistant with the provided configuration"""
    global assistant_thread, assistant_process, is_running
    
    if is_running:
        return JSONResponse(
            status_code=400,
            content={"error": "Assistant is already running. Stop it first."}
        )
    
    try:
        use_thread = True 
        
        if use_thread:
            assistant_thread = threading.Thread(
                target=run_assistant_in_thread,
                args=(config,),
                daemon=True
            )
            assistant_thread.start()
        else:
            background_tasks.add_task(run_assistant_subprocess, config)
        
        return {
            "status": "started",
            "config": config.dict(),
            "message": "Cursor Assistant is starting. Check /status for updates."
        }
    
    except Exception as e:
        import traceback
        error_msg = f"Failed to start assistant: {str(e)}\n{traceback.format_exc()}"
        return JSONResponse(
            status_code=500,
            content={"error": error_msg}
        )


@app.post("/stop", response_class=JSONResponse)
async def stop_assistant():
    """Stop the running Cursor Assistant"""
    global assistant_process, assistant_instance, is_running
    
    if not is_running:
        return JSONResponse(
            status_code=400,
            content={"error": "Assistant is not running."}
        )
    
    try:
        if assistant_instance is not None:
            assistant_instance.is_running = False
            if hasattr(assistant_instance.recorder, 'stop_recording'):
                assistant_instance.recorder.stop_recording()
        
        if assistant_process is not None:
            assistant_process.terminate()
            
            for _ in range(5):
                if assistant_process.poll() is not None:
                    break
                time.sleep(0.5)
            
            if assistant_process.poll() is None:
                assistant_process.kill()
        
        is_running = False
        
        return {"status": "stopped", "message": "Cursor Assistant has been stopped."}
    
    except Exception as e:
        import traceback
        error_msg = f"Failed to stop assistant: {str(e)}\n{traceback.format_exc()}"
        return JSONResponse(
            status_code=500,
            content={"error": error_msg}
        )


@app.get("/status", response_model=AssistantStatus)
async def get_status():
    """Get the current status of the Cursor Assistant"""
    global assistant_process, is_running
    
    status = AssistantStatus(running=is_running)
    
    if is_running:
        if assistant_process is not None:
            pid = assistant_process.pid
            status.pid = pid
            
            try:
                process = psutil.Process(pid)
                status.uptime = time.time() - process.create_time()
                status.start_time = time.strftime(
                    "%Y-%m-%d %H:%M:%S", 
                    time.localtime(process.create_time())
                )
                status.memory_usage = f"{process.memory_info().rss / (1024 * 1024):.2f} MB"
                status.cpu_percent = process.cpu_percent(interval=0.1)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        elif assistant_thread is not None and assistant_thread.is_alive():
            status.pid = os.getpid() 
    
    return status


@app.get("/logs")
async def get_logs(
    lines: int = Query(50, description="Number of log lines to return", ge=1, le=1000),
    full: bool = Query(False, description="Return all available logs")
):
    """Get the logs from the assistant process"""
    global process_output
    
    if full:
        return {"logs": process_output}
    else:
        return {"logs": process_output[-lines:] if process_output else []}


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok"}


def start_ngrok():
    """Start ngrok to expose the server to the internet"""
    try:
        from pyngrok import ngrok
        
        ngrok_auth_token = os.environ.get("NGROK_AUTH_TOKEN")
        if ngrok_auth_token:
            ngrok.set_auth_token(ngrok_auth_token)
        
        public_url = ngrok.connect(8000).public_url
        print(f"ngrok tunnel opened at: {public_url}")
        
    except ImportError:
        print("ngrok not installed. Run 'pip install pyngrok' to enable tunneling.")
    except Exception as e:
        print(f"Failed to start ngrok: {str(e)}")


if __name__ == "__main__":
    threading.Thread(target=start_ngrok, daemon=True).start()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)