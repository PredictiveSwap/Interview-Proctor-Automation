import os
import time
import json
import threading
import speech_recognition as sr
import pyttsx3
import pygame
import subprocess
from dotenv import load_dotenv
from intrusion_detector import IntrusionDetector

# Load environment variables
load_dotenv()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Initialize pygame for timer sounds
pygame.init()
pygame.mixer.init()

# Load role type lists
TECH_ROLES = set(open('tech.txt').read().strip().split(', '))
NON_TECH_ROLES = set(open('non-tech.txt').read().strip().split(', '))

def call_ollama(prompt, model_name):
    """Call Ollama locally using subprocess"""
    try:
        # Construct the Ollama command
        cmd = ["ollama", "run", model_name, prompt]
        
        # Run Ollama command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Return the output if successful
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"Error from Ollama: {result.stderr}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

class InterviewConductor:
    def __init__(self):
        self.job_role = ""
        self.interview_duration = 0  # in minutes
        self.answer_time = 0  # in seconds
        self.num_questions = 0
        self.interview_questions = []
        self.candidate_answers = []
        self.current_question_index = 0
        self.interview_in_progress = False
        self.timer_thread = None
        self.recording_thread = None
        self.stop_recording = False
        self.is_tech_role = False
        
        # Initialize intrusion detector
        self.intrusion_detector = IntrusionDetector()
        self.use_intrusion_detection = False

    def is_technical_role(self, role):
        """Check if the given role is technical"""
        return any(tech_role.lower() in role.lower() for tech_role in TECH_ROLES)

    def is_non_technical_role(self, role):
        """Check if the given role is non-technical"""
        return any(non_tech_role.lower() in role.lower() for non_tech_role in NON_TECH_ROLES)

    def setup_interview(self):
        """Set up the interview parameters"""
        print("\n===== INTERVIEW SETUP =====")
        self.job_role = input("Enter the job role for the interview: ")
        
        # Determine if it's a technical role
        self.is_tech_role = self.is_technical_role(self.job_role)
        if not self.is_tech_role and not self.is_non_technical_role(self.job_role):
            print("Warning: Role type not recognized. Defaulting to non-technical.")
            self.is_tech_role = False
        
        print(f"Role type: {'Technical' if self.is_tech_role else 'Non-technical'}")
        print(f"Using model: {'deepseek-coder:6.7b' if self.is_tech_role else 'llama2:8b'}")
        
        while True:
            try:
                self.interview_duration = int(input("Enter the interview duration (in minutes): "))
                if self.interview_duration <= 0:
                    print("Please enter a positive number.")
                    continue
                break
            except ValueError:
                print("Please enter a valid number.")
        
        while True:
            try:
                self.answer_time = int(input("Enter the time allowed for each answer (in seconds): "))
                if self.answer_time <= 0:
                    print("Please enter a positive number.")
                    continue
                break
            except ValueError:
                print("Please enter a valid number.")
        
        # Ask if intrusion detection should be enabled
        while True:
            response = input("Enable intrusion detection? (y/n): ").lower()
            if response in ['y', 'yes']:
                self.use_intrusion_detection = True
                break
            elif response in ['n', 'no']:
                self.use_intrusion_detection = False
                break
            else:
                print("Please enter 'y' or 'n'.")
        
        # Set up intrusion detection if enabled
        if self.use_intrusion_detection:
            print("\n===== INTRUSION DETECTION SETUP =====")
            print("We'll now capture a reference image of your face.")
            print("This will be used to detect any intrusions during the interview.")
            
            if not self.intrusion_detector.capture_reference_face():
                print("Failed to capture reference face. Intrusion detection will be disabled.")
                self.use_intrusion_detection = False
        
        # Calculate the number of questions based on interview duration and answer time
        # Allow for some buffer time between questions (15 seconds per question for transitions)
        buffer_time_per_question = 15  # seconds
        total_available_seconds = self.interview_duration * 60
        time_per_question = self.answer_time + buffer_time_per_question
        
        self.num_questions = max(1, min(10, int(total_available_seconds / time_per_question)))
        
        print(f"\nBased on your settings:")
        print(f"- Interview duration: {self.interview_duration} minutes")
        print(f"- Time per answer: {self.answer_time} seconds")
        print(f"- The interview will include {self.num_questions} questions")
        print(f"- Intrusion detection: {'Enabled' if self.use_intrusion_detection else 'Disabled'}")
        
        print(f"\nGenerating interview questions for {self.job_role} role...")
        self.generate_questions()
        print(f"Generated {len(self.interview_questions)} questions for the interview.")
        
        print("\nInterview setup complete! Press Enter to start the interview...")
        input()
        self.start_interview()

    def generate_questions(self):
        """Generate interview questions using Ollama"""
        try:
            # Select the appropriate model based on role type
            model_name = "deepseek-coder:6.7b" if self.is_tech_role else "llama2:8b"
            
            # Prepare the prompt
            prompt = f"""Generate exactly {self.num_questions} interview questions for a {self.job_role} position.
The interview will last {self.interview_duration} minutes total, with {self.answer_time} seconds per answer.
Format your response as a numbered list of questions.
Make the questions specific to the role and its requirements.
For technical roles, include specific technical questions related to the field."""

            # Call Ollama API
            response = call_ollama(prompt, model_name)
            
            if response:
                # Process the response to extract questions
                lines = response.strip().split('\n')
                questions = []
                
                for line in lines:
                    # Remove common prefixes and clean up the line
                    line = line.strip()
                    for prefix in ['- ', '* ', '. ', '• ', '1. ', '2. ', '3. ', '4. ', '5. ', '6. ', '7. ', '8. ', '9. ', '10. ']:
                        if line.startswith(prefix):
                            line = line[len(prefix):].strip()
                    
                    if line and not line.lower().startswith(('here', 'question', 'following')):
                        questions.append(line)
                
                # Ensure we have exactly the calculated number of questions
                if questions:
                    if len(questions) > self.num_questions:
                        self.interview_questions = questions[:self.num_questions]
                    elif len(questions) < self.num_questions:
                        # Add generic questions if we don't have enough
                        generic_questions = [
                            "Tell me about yourself and your experience.",
                            f"Why are you interested in this {self.job_role} position?",
                            "What are your greatest strengths and weaknesses?",
                            "Describe a challenging situation you faced and how you handled it.",
                            "Where do you see yourself in five years?",
                            "What do you know about our company?",
                            "How do you handle stress and pressure?",
                            "What is your greatest professional achievement?",
                            "How would your colleagues describe you?",
                            "Do you have any questions for us?"
                        ]
                        
                        # Add generic questions until we reach the desired number
                        needed = self.num_questions - len(questions)
                        self.interview_questions = questions + generic_questions[:needed]
                    else:
                        self.interview_questions = questions
                else:
                    raise Exception("No valid questions generated")
            else:
                raise Exception("No response from Ollama API")
            
            # Initialize candidate answers list with empty strings
            self.candidate_answers = [""] * len(self.interview_questions)
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            # Fallback to generic questions
            generic_questions = [
                "Tell me about yourself and your experience.",
                f"Why are you interested in this {self.job_role} position?",
                "What are your greatest strengths and weaknesses?",
                "Describe a challenging situation you faced and how you handled it.",
                "Where do you see yourself in five years?",
                "What do you know about our company?",
                "How do you handle stress and pressure?",
                "What is your greatest professional achievement?",
                "How would your colleagues describe you?",
                "Do you have any questions for us?"
            ]
            
            # Use only as many questions as calculated
            self.interview_questions = generic_questions[:self.num_questions]
            self.candidate_answers = [""] * len(self.interview_questions)

    def speak(self, text):
        """Convert text to speech"""
        print(f"Interviewer: {text}")
        engine.say(text)
        engine.runAndWait()

    def play_sound(self, sound_type):
        """Play a sound to indicate timer events"""
        beep_frequency = 440  # A4 note
        beep_duration = 0.5  # seconds
        
        if sound_type == "start":
            beep_frequency = 523  # C5 note - higher pitch for start
        elif sound_type == "warning":
            beep_frequency = 392  # G4 note - middle pitch for warning
        elif sound_type == "end":
            beep_frequency = 262  # C4 note - lower pitch for end
        elif sound_type == "intrusion":
            beep_frequency = 660  # E5 note - higher pitch for intrusion alert
        
        sample_rate = 44100
        samples = pygame.sndarray.make_sound(
            (32767 * pygame.sndarray.array(
                [pygame.math.sin(2 * 3.14159 * beep_frequency * t / sample_rate) 
                 for t in range(int(beep_duration * sample_rate))]
            )).astype(pygame.sndarray.dtype))
        
        samples.play()
        time.sleep(beep_duration)

    def timer_countdown(self, seconds):
        """Countdown timer with warnings"""
        self.play_sound("start")
        print(f"\nTimer started: {seconds} seconds remaining")
        
        warning_threshold = max(5, int(seconds * 0.2))  # Warning at 20% time left or 5 seconds, whichever is greater
        
        for remaining in range(seconds, 0, -1):
            if not self.interview_in_progress:
                break
                
            if remaining == warning_threshold:
                self.play_sound("warning")
                print(f"\nWarning: {remaining} seconds remaining")
                
            time.sleep(1)
            
        if self.interview_in_progress:
            self.play_sound("end")
            print("\nTime's up!")

    def record_audio(self, question_index):
        """Record the candidate's answer"""
        self.stop_recording = False
        with sr.Microphone() as source:
            print("\nListening to your answer...")
            audio_data = []
            
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Record audio until timer ends or interview stops
            start_time = time.time()
            while time.time() - start_time < self.answer_time and not self.stop_recording and self.interview_in_progress:
                try:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=10)
                    audio_data.append(audio)
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"Error recording audio: {e}")
                    break
            
            # Process recorded audio
            if audio_data and self.interview_in_progress:
                full_text = ""
                for audio in audio_data:
                    try:
                        text = recognizer.recognize_google(audio)
                        full_text += " " + text
                    except sr.UnknownValueError:
                        print("Could not understand audio")
                    except sr.RequestError as e:
                        print(f"Error with speech recognition service: {e}")
                
                self.candidate_answers[question_index] = full_text.strip()
                print(f"\nYour answer: {self.candidate_answers[question_index]}")
            else:
                self.candidate_answers[question_index] = "[No answer recorded]"

    def start_interview(self):
        """Start the interview process"""
        self.interview_in_progress = True
        self.current_question_index = 0
        
        # Start intrusion detection if enabled
        if self.use_intrusion_detection:
            if not self.intrusion_detector.start_monitoring():
                print("Failed to start intrusion detection. Continuing without it.")
                self.use_intrusion_detection = False
        
        self.speak("Welcome to your automated interview session.")
        self.speak(f"This interview is for the {self.job_role} position and will last approximately {self.interview_duration} minutes.")
        self.speak(f"You will have {self.answer_time} seconds to answer each question.")
        self.speak(f"There will be {len(self.interview_questions)} questions in total.")
        
        if self.use_intrusion_detection:
            self.speak("Please note that intrusion detection is enabled. Please stay within the camera frame and avoid any external assistance.")
        
        self.speak("Let's begin with the first question.")
        
        start_time = time.time()
        
        while self.current_question_index < len(self.interview_questions) and self.interview_in_progress:
            question = self.interview_questions[self.current_question_index]
            
            # Ask the question
            self.speak(f"Question {self.current_question_index + 1}: {question}")
            
            # Start timer and recording in separate threads
            self.timer_thread = threading.Thread(target=self.timer_countdown, args=(self.answer_time,))
            self.recording_thread = threading.Thread(target=self.record_audio, args=(self.current_question_index,))
            
            self.timer_thread.start()
            self.recording_thread.start()
            
            # Wait for both threads to complete
            self.timer_thread.join()
            self.stop_recording = True
            self.recording_thread.join()
            
            # Check intrusion status if enabled
            if self.use_intrusion_detection and self.intrusion_detector.intrusion_count > 0:
                intrusion_status = self.intrusion_detector.get_intrusion_status()
                if intrusion_status["intrusion_count"] > 0:
                    self.play_sound("intrusion")
                    self.speak(f"Warning: {intrusion_status['intrusion_count']} intrusions detected during your answer. Please ensure you are not receiving external assistance.")
            
            # Move to the next question
            self.current_question_index += 1
            
            # Check if we're running out of time
            elapsed_minutes = (time.time() - start_time) / 60
            remaining_minutes = self.interview_duration - elapsed_minutes
            
            # If less than 2 minutes remaining and we still have more than one question left, adjust
            if remaining_minutes < 2 and self.current_question_index < len(self.interview_questions) - 1:
                skipped = len(self.interview_questions) - self.current_question_index - 1
                self.speak(f"We're running short on time. Skipping {skipped} questions to reach the final question.")
                self.current_question_index = len(self.interview_questions) - 1
            
            if self.current_question_index < len(self.interview_questions):
                self.speak("Thank you for your answer. Moving to the next question.")
            else:
                self.speak("Thank you for completing all the questions in this interview.")
        
        self.end_interview()

    def end_interview(self):
        """End the interview and provide feedback"""
        self.interview_in_progress = False
        
        # Stop intrusion detection if it was enabled
        if self.use_intrusion_detection:
            intrusion_status = self.intrusion_detector.get_intrusion_status()
            self.intrusion_detector.stop_monitoring()
        
        if self.current_question_index >= len(self.interview_questions):
            self.speak("The interview is now complete.")
            
            print("\n===== INTERVIEW SUMMARY =====")
            for i, (question, answer) in enumerate(zip(self.interview_questions, self.candidate_answers)):
                print(f"\nQuestion {i+1}: {question}")
                print(f"Your answer: {answer}")
            
            # Report on intrusions if detection was enabled
            if self.use_intrusion_detection:
                print(f"\nIntrusion detection report:")
                print(f"- Total intrusions detected: {intrusion_status['intrusion_count']}")
                if intrusion_status['intrusion_count'] > 0:
                    print(f"- Intrusion images saved as: intrusion_1.jpg, intrusion_2.jpg, etc.")
            
            print("\nGenerating feedback...")
            self.generate_feedback()
        else:
            self.speak("The interview has been terminated early.")
    
    def generate_feedback(self):
        """Generate feedback using Ollama"""
        try:
            # Select the appropriate model based on role type
            model_name = "deepseek-coder:6.7b" if self.is_tech_role else "llama2:8b"
            
            # Prepare the interview data
            interview_summary = "Interview Summary:\n\n"
            for i, (question, answer) in enumerate(zip(self.interview_questions, self.candidate_answers)):
                interview_summary += f"Question {i+1}: {question}\n"
                interview_summary += f"Answer: {answer if answer else '[No answer provided]'}\n\n"
            
            # Add intrusion information if available
            if self.use_intrusion_detection:
                intrusion_status = self.intrusion_detector.get_intrusion_status()
                interview_summary += f"\nNote: During the interview, {intrusion_status['intrusion_count']} potential intrusions were detected."
            
            # Prepare the prompt
            prompt = f"""You are an expert interviewer providing constructive feedback for a {self.job_role} position.
Please analyze the following interview and provide detailed feedback:

{interview_summary}

Provide feedback in the following format:
1. Overall Assessment
2. Key Strengths
3. Areas for Improvement
4. Final Recommendations"""
            
            # Call Ollama API
            feedback = call_ollama(prompt, model_name)
            
            if feedback:
                print("\n===== INTERVIEW FEEDBACK =====")
                print(feedback)
                self.speak("I've generated feedback based on your interview. You can review it in the console.")
            else:
                raise Exception("No response from Ollama API")
            
        except Exception as e:
            print(f"Error generating feedback: {e}")
            print("\nUnable to generate automated feedback. Please review the interview summary.")

    def stop_interview(self):
        """Stop the ongoing interview"""
        if self.interview_in_progress:
            self.interview_in_progress = False
            self.stop_recording = True
            
            # Stop intrusion detection if it was enabled
            if self.use_intrusion_detection:
                self.intrusion_detector.stop_monitoring()
            
            print("\nStopping the interview...")
            time.sleep(1)  # Give threads time to clean up
            self.speak("The interview has been stopped.")

def main():
    print("===== AUTOMATED INTERVIEW CONDUCTOR =====")
    print("This program will conduct an automated interview using OpenAI's API.")
    
    interviewer = InterviewConductor()
    
    try:
        interviewer.setup_interview()
    except KeyboardInterrupt:
        print("\nInterview setup interrupted.")
        interviewer.stop_interview()
    
    print("\nThank you for using the Automated Interview Conductor!")
    print("Press Ctrl+C to exit the program.")
    
    try:
        # Keep the program running until explicitly terminated
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting the program...")
        pygame.quit()

if __name__ == "__main__":
    main() 