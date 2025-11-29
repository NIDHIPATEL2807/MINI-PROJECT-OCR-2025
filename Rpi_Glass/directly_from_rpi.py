from picamera2 import Picamera2
import RPi.GPIO as GPIO
import time
import requests
import json
from gtts import gTTS
import os

# ---------------- CONFIG ---------------- #
BUTTON_PIN = 17
FLASK_SERVER_URL = "http://127.0.0.1:5000/ocr"
SAVE_PATH = "/home/aayush/smart_ocr_rpi/captures/"
AUDIO_SAVE_PATH = "/home/aayush/smart_ocr_rpi/audio/"

# Create folders
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(AUDIO_SAVE_PATH, exist_ok=True)

# ---------------- GPIO SETUP ---------------- #
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# ---------------- CAMERA SETUP ---------------- #
camera = Picamera2()
camera.configure(camera.create_still_configuration())

print("📸 Ready! Press the button to capture and send to OCR server...")

try:
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.HIGH:
            time.sleep(0.3)  # debounce
            print("🔘 Button pressed! Capturing photo...")

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            img_name = f"img_{timestamp}.jpg"
            img_path = os.path.join(SAVE_PATH, img_name)

            # Capture image
            camera.start()
            time.sleep(1)
            camera.capture_file(img_path)
            camera.stop()

            print(f"📁 Image saved: {img_path}")

            # ---------------- SEND TO OCR ---------------- #
            try:
                payload = {"image_path": img_path}
                headers = {"Content-Type": "application/json"}

                print("📤 Sending to local OCR server...")
                response = requests.post(FLASK_SERVER_URL, data=json.dumps(payload), headers=headers)

                if response.ok:
                    extracted_text = response.json().get("text", "").strip()

                    if extracted_text:
                        print("🧠 OCR Text:", extracted_text)

                        # -------- TTS -------- #
                        audio_file = f"speech_{timestamp}.mp3"
                        audio_path = os.path.join(AUDIO_SAVE_PATH, audio_file)

                        tts = gTTS(text=extracted_text, lang="hi")
                        tts.save(audio_path)

                        print(f"🔊 Audio saved: {audio_path}")
                    else:
                        print("⚠️ No text returned.")

                else:
                    print("❌ OCR server error:", response.text)

            except Exception as e:
                print("⚠️ Failed to reach OCR server:", e)

            time.sleep(1)

except KeyboardInterrupt:
    print("\n👋 Exiting...")

finally:
    camera.close()
    GPIO.cleanup()
