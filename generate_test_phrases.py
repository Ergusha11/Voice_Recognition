import os
import random
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment

# Configuration
OUTPUT_DIR = "synthetic_phrases_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define logical groups (Same as training, but we generate NEW audio instances)
VERBS = ["ir", "alto"]
DIRECTIONS = ["adelante", "atrás", "izquierda", "derecha"]
NUMBERS = ["uno", "dos", "tres", "cuatro", "cinco"]
CONFIRMATION = ["sí", "no"]

phrases = []
for v in VERBS:
    for d in DIRECTIONS:
        phrases.append(f"{v} {d}")
for n in NUMBERS:
    for d in DIRECTIONS:
        phrases.append(f"{n} {d}")
phrases.extend(DIRECTIONS)
phrases.extend(CONFIRMATION)

# Generate only 1 variation per phrase for testing
NUM_VARIATIONS = 1 

def speed_change(sound, speed=1.0):
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

print(f"Generating TEST audio into '{OUTPUT_DIR}'...")

for i, phrase in enumerate(phrases):
    safe_filename = phrase.replace(" ", "_")
    for j in range(NUM_VARIATIONS):
        try:
            # Randomize TLD and Speed again to ensure uniqueness
            tld = random.choice(['com.mx', 'es', 'us'])
            speed = random.uniform(0.8, 1.3) # Slightly wider range for testing robustness
            
            tts = gTTS(text=phrase, lang='es', tld=tld, slow=False)
            mp3_fp = BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            
            base_audio = AudioSegment.from_file(mp3_fp, format="mp3")
            aug_audio = speed_change(base_audio, speed)

            filename = f"{safe_filename}_test_{i}.wav"
            aug_audio.export(os.path.join(OUTPUT_DIR, filename), format="wav")
            
        except Exception as e:
            print(f"Error: {e}")

print("Test data generation complete.")