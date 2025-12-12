import os
import random
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment

# Output directory for phrases
OUTPUT_DIR = "synthetic_phrases"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define logical groups
VERBS = ["ir", "alto"]
DIRECTIONS = ["adelante", "atrás", "izquierda", "derecha"]
NUMBERS = ["uno", "dos", "tres", "cuatro", "cinco"]
CONFIRMATION = ["sí", "no"]

# Generate coherent combinations
phrases = []

# 1. Action + Direction (e.g., "ir izquierda")
for v in VERBS:
    for d in DIRECTIONS:
        phrases.append(f"{v} {d}")

# 2. Number + Direction (e.g., "dos adelante")
for n in NUMBERS:
    for d in DIRECTIONS:
        phrases.append(f"{n} {d}")

# 3. Just Directions (e.g., "izquierda")
phrases.extend(DIRECTIONS)

# 4. Confirmations (simulating simple answers)
phrases.extend(CONFIRMATION)

print(f"Defined {len(phrases)} unique phrase templates.")

NUM_VARIATIONS = 5 # Variations per phrase

def speed_change(sound, speed=1.0):
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

print(f"Generating synthetic audio into '{OUTPUT_DIR}'...")

for i, phrase in enumerate(phrases):
    # Sanitize filename
    safe_filename = phrase.replace(" ", "_")
    
    for j in range(NUM_VARIATIONS):
        try:
            # Random accent
            tld = random.choice(['com.mx', 'es', 'us'])
            
            tts = gTTS(text=phrase, lang='es', tld=tld, slow=False)
            mp3_fp = BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            
            base_audio = AudioSegment.from_file(mp3_fp, format="mp3")

            # Random speed/pitch
            speed = random.uniform(0.85, 1.25)
            aug_audio = speed_change(base_audio, speed)

            # Save: "phrase_variation.wav"
            # Format: textlabel@index_variation.wav 
            # We use '@' as a separator to easily extract the text label later
            filename = f"{safe_filename}@{i}_{j}.wav"
            aug_audio.export(os.path.join(OUTPUT_DIR, filename), format="wav")
            
        except Exception as e:
            print(f"Error: {e}")

print("Phrase generation complete.")