import pywhatkit
import speech_recognition as sr
import datetime
import pyttsx3  # Optional: for voice feedback

def speak(text):
    # Optional function to provide voice feedback
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio)
        print("You said:", command)
        return command.lower()
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        speak("Sorry, I did not understand that.")
        return ""
    except sr.RequestError:
        print("Network issue. Please check your connection.")
        speak("Network issue. Please check your connection.")
        return ""

import pywhatkit
import datetime

def process_command(command):
    command = command.lower()
    
    if "play" in command:
        song = command.split("play", 1)[1].strip()
        if song:
            print(f"Playing {song} on YouTube...")
            pywhatkit.playonyt(song)

    elif "search" in command:
        query = command.split("search", 1)[1].strip()
        if query:
            print(f"Searching for {query} on Google...")
            pywhatkit.search(query)

    elif "message" in command:
        try:
            # Extract number and message
            parts = command.split("message", 1)[1].strip()

            # Assume user says something like: "972-809-4223 hello there"
            split_parts = parts.split(" ", 1)
            raw_number = split_parts[0].replace("-", "").replace(" ", "")
            
            if not raw_number.startswith("+"):
                # Add default country code if not provided
                phone = "+1" + raw_number
            else:
                phone = raw_number

            # Get message if provided
            message = split_parts[1] if len(split_parts) > 1 else "Hello from Nathan's assistant!"

            # Get current time
            now = datetime.datetime.now() + datetime.timedelta(minutes=1)
            hour = now.hour
            minute = now.minute

            print(f"Sending WhatsApp message to {phone} at {hour}:{minute}")
            pywhatkit.sendwhatmsg(phone, message, hour, minute)
        
        except Exception as e:
            print("Error sending WhatsApp message:", e)

    else:
        print("Command not recognized or supported.")


def main():
    command = listen_command()
    if command:
        process_command(command)
    else:
        print("No valid command received.")

if __name__ == "__main__":
    main()
