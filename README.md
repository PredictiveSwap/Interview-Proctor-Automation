# Automated Interview Conductor

An intelligent automated interview system that conducts interviews using local Large Language Models (LLMs) through Ollama. The system features speech recognition, text-to-speech, intrusion detection, and automated feedback generation.

## Features

- **Intelligent Question Generation**: Uses Ollama models to generate role-specific interview questions
  - Technical roles: Uses `deepseek-coder:6.7b` for specialized technical questions
  - Non-technical roles: Uses `llama2:8b` for general and behavioral questions
- **Speech Recognition**: Converts candidate's spoken answers to text
- **Text-to-Speech**: Provides clear, spoken questions and instructions
- **Intrusion Detection**: Optional camera-based monitoring to ensure interview integrity
- **Automated Feedback**: Generates comprehensive feedback using role-appropriate LLM
- **Timed Responses**: Manages interview duration with configurable answer time limits
- **Progress Monitoring**: Tracks interview progress with audio cues and visual feedback

## Prerequisites

1. Python 3.10 or higher
2. Ollama installed on your system
3. Required Ollama models:
   - `deepseek-coder:6.7b` (for technical interviews)
   - `llama2:8b` (for non-technical interviews)
4. Webcam (optional, for intrusion detection)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/automated-interview-conductor.git
cd automated-interview-conductor
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Install Ollama models:
```bash
ollama pull deepseek-coder:6.7b
ollama pull llama2:8b
```

## Usage

1. Start the program:
```bash
python interview_conductor.py
```

2. Follow the setup prompts:
   - Enter the job role
   - Specify interview duration
   - Set time allowed for each answer
   - Choose whether to enable intrusion detection

3. If intrusion detection is enabled:
   - Position yourself in front of the camera
   - A reference image will be captured
   - Stay within frame during the interview

4. The interview will begin:
   - Listen to each question
   - Respond verbally within the time limit
   - Audio cues will indicate start, warnings, and end of response time

5. After completion:
   - Review the interview summary
   - Get AI-generated feedback
   - Check intrusion detection report (if enabled)

## Role Classification

The system automatically classifies roles as technical or non-technical using predefined lists:
- Technical roles use `deepseek-coder:6.7b` for specialized questions
- Non-technical roles use `llama2:8b` for general questions

## Dependencies

- `speech_recognition`: For converting speech to text
- `pyttsx3`: For text-to-speech conversion
- `pygame`: For audio cues
- `opencv-python`: For intrusion detection
- `numpy`: For image processing
- `Pillow`: For image handling

## Notes

- Ensure your microphone and speakers are working properly
- Good lighting is recommended for intrusion detection
- Internet connection is required for speech recognition
- Local Ollama installation must be running during the interview

## Troubleshooting

1. If speech recognition fails:
   - Check your microphone settings
   - Ensure you have a stable internet connection

2. If intrusion detection isn't working:
   - Check your webcam connection
   - Ensure proper lighting
   - Verify OpenCV installation

3. If Ollama models aren't responding:
   - Verify Ollama is running on your system
   - Check if models are properly installed
   - Run `ollama list` to verify available models

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 