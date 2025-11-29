from gtts import gTTS
import sys

def text_to_audio(text, filename="static/output.mp3"):
    tts = gTTS(text=text, lang="hi")
    tts.save(filename)
    print("Audio saved at:", filename)

if __name__ == "__main__":
    text_input = sys.argv[1]     # Read text from command line
    text_to_audio(text_input)
