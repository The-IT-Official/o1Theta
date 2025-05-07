import pyttsx3
import speech_recognition as sr

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()

# Set voice directly using index (e.g., voices[7].id)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[14].id)  # You can change the index to test other voices

engine.setProperty('rate', 150)  # Speaking rate

# Jarvis speaks
def jarvis_speak(text):
    print(f"JARVIS: {text}")
    engine.say(text)
    engine.runAndWait()

# Listen to your voice
def listen_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.pause_threshold = 1
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        query = recognizer.recognize_google(audio)
        print(f"You said: {query}")
    except:
        jarvis_speak("Sorry, I didn't catch that.")
        return ""
    return query

# Main loop
jarvis_speak("Welcome back, Nathan. How may I assist you today?")

while True:
    command = listen_command().lower()
    if "exit" in command or "stop" in command:
        jarvis_speak("Goodbye, sir. Shutting down.")
        break
    elif command:
        jarvis_speak(f"You said: {command}")
