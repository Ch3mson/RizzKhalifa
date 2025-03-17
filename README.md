# RizzKhalifa

A conversational AI assistant designed to help you sound smooth and charismatic in dating conversations. The assistant listens to your conversations, analyzes the context, and provides witty, thoughtful responses that make you sound impressive.

## Features

- **Active Listening Mode**: Automatically provides response suggestions when your date speaks
- **Speech Recognition**: Uses advanced speech recognition with speaker diarization to identify different speakers
- **Smooth Responses**: Generates charismatic, contextually relevant responses
- **User Knowledge Base**: Incorporates your personal information to make responses more authentic
- **Voice Generation**: Converts responses to speech with natural-sounding voices
- **Facial Recognition**: Optional feature to detect and identify faces during conversations

## Requirements

- Python 3.8+
- OpenAI API key (for voice generation and embeddings)
- Groq API key (for text generation)
- Microphone access
- Camera access (optional, for facial recognition)

## Dependencies

```
openai>=1.0.0
groq>=0.4.0
tiktoken>=0.5.0
numpy>=1.22.0
nltk>=3.8.1
python-dotenv>=1.0.0
supabase>=1.0.0 (optional, for storage)
opencv-python>=4.6.0 (optional, for facial recognition)
pyautogui>=0.9.53 (optional, for screen capture)
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/https://github.com/Ch3mson/RizzKhalifa.git
   cd meta_rizz
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   
   Create a `.env` file in the project root with the following variables:
   ```
   LANGSMITH_TRACING=true
   LANGSMITH_ENDPOINT=your_langsmit_endpoint
   LANGSMITH_API_KEY=your_langsmith_api_key
   LANGSMITH_PROJECT=your_langsmith_project
   OPENAI_API_KEY=your_openai_api_key
   GROQ_API_KEY=your_groq_api_key
   SUPABASE_URL=your_supabase_url (optional)
   SUPABASE_KEY=your_supabase_key (optional)
   ```

## Usage

### Running the Assistant

The application has two main components that need to run simultaneously:
- `cursor_main.py` - The conversation assistant that listens and generates responses
- `main.py` - The primary application (referred to in cursor_main.py)

To run the application:

1. **Start the main application**:
   ```bash
   python ./api_server.py
   ```

2. **In a separate terminal, start the cursor assistant**:
   ```bash
   python cursor_main.py
   ```

### Command-Line Arguments

The cursor assistant supports several command-line options:

- `--diarization`: Enable speaker diarization (default: enabled)
- `--no-diarization`: Disable speaker diarization
- `--speakers N`: Set the expected number of speakers (default: 2)
- `--screen`: Enable screen capture for facial recognition (default: disabled)
- `--debug`: Enable debug mode with verbose logging

Example with options:
```bash
python cursor_main.py --no-diarization --screen --debug
```

## Usage Steps

1. **Start both applications** as described above
2. **Voice Reference**: If diarization is enabled, you'll be prompted to provide a 10-second voice sample
3. **Activate Listening Mode**: Say the trigger phrase (default: "hey cursor") to activate active listening mode
4. **Conversation Flow**:
   - The assistant will listen to the conversation
   - When your date speaks, the assistant will analyze their message
   - After a short cooldown period (15 seconds), the assistant will suggest a smooth response
   - You can say "let me think" at any time to get an immediate response suggestion
5. **Deactivate Listening Mode**: Say the stop phrase (default: "stop cursor") to exit active listening mode

6. **Facial Recognition (optional)**:
7. If using Meta glasses, call via messenger to allow faces to be captured through the computer screen. This allows the app to use --screen for capturing faces on your device

## Troubleshooting

- **No Audio Detected**: Check your microphone settings and permissions
- **Speech Recognition Issues**: Ensure you're in a quiet environment with clear speech
- **API Key Errors**: Verify your OpenAI and Groq API keys in the .env file
- **Facial Recognition Issues**: Check camera permissions and lighting conditions
- **Activation Phrase Not Detected**: Speak clearly and directly when saying the activation phrase

## License

This project is licensed under the MIT License - see the LICENSE file for details.
