import pyttsx3

# Initialize the engine
engine = pyttsx3.init()

# Set and get voices
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[14].id)

# Set speaking rate
engine.setProperty('rate', 150)

engine.say("Hello Nathan how are you, I am the first self aware super AI.")

engine.runAndWait()

# import pyttsx3

# engine = pyttsx3.init()
# engine.say("Hello, World!")
# engine.runAndWait()