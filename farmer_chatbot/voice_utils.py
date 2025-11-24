"""
Voice utilities for Speech-to-Text (STT) and Text-to-Speech (TTS)
"""
import speech_recognition as sr
import pyttsx3
import io
import os
import tempfile
from gtts import gTTS
import base64

class VoiceAssistant:
    def __init__(self):
        """Initialize voice recognition and text-to-speech engines."""
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.tts_engine = None
        self.use_online_tts = True  # Use gTTS (online) by default
        
        # Try to initialize TTS engine (offline)
        try:
            self.tts_engine = pyttsx3.init()
            self.use_online_tts = False
            # Set speech rate (words per minute)
            self.tts_engine.setProperty('rate', 150)
            # Set volume (0.0 to 1.0)
            self.tts_engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"Offline TTS not available, will use online TTS: {e}")
            self.use_online_tts = True
        
        # Try to initialize microphone
        try:
            self.microphone = sr.Microphone()
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception as e:
            print(f"Microphone not available: {e}")
            self.microphone = None
    
    def listen(self, timeout=5, phrase_time_limit=10):
        """
        Listen to microphone input and convert speech to text.
        
        Args:
            timeout: Maximum time to wait for speech to start
            phrase_time_limit: Maximum time for a phrase
            
        Returns:
            Recognized text or None if failed
        """
        if self.microphone is None:
            return None
        
        try:
            with self.microphone as source:
                # Listen for audio input
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
            
            # Try Google Speech Recognition first (online, better accuracy)
            try:
                text = self.recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                return None
            except sr.RequestError:
                # Fallback to offline recognition
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    return text
                except:
                    return None
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return None
    
    def speak(self, text, lang='en'):
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to speak
            lang: Language code (default: 'en' for English)
        """
        if not text:
            return
        
        if self.use_online_tts:
            # Use Google Text-to-Speech (online, better quality)
            self._speak_online(text, lang)
        else:
            # Use offline TTS
            self._speak_offline(text)
    
    def _speak_offline(self, text):
        """Speak using offline pyttsx3 engine."""
        try:
            if self.tts_engine:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Error in offline TTS: {e}")
    
    def _speak_online(self, text, lang='en'):
        """Generate speech using gTTS and return audio file path."""
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tts.save(temp_file.name)
            return temp_file.name
        except Exception as e:
            print(f"Error in online TTS: {e}")
            return None
    
    def text_to_audio_file(self, text, lang='en'):
        """
        Convert text to speech and return audio file path.
        Useful for Streamlit audio playback.
        
        Args:
            text: Text to convert
            lang: Language code
            
        Returns:
            Path to audio file or None
        """
        if not text:
            return None
        
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tts.save(temp_file.name)
            return temp_file.name
        except Exception as e:
            print(f"Error generating audio file: {e}")
            return None


# Browser-based STT using Web Speech API (for Streamlit)
def get_audio_recorder_html():
    """Generate HTML for browser-based audio recording."""
    return """
    <script>
    let mediaRecorder;
    let audioChunks = [];
    
    async function startRecording() {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            // You would send this to your backend for processing
            console.log('Recording stopped', audioBlob);
        };
        
        mediaRecorder.start();
    }
    
    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
    }
    </script>
    """

