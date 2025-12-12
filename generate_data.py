import os
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
import random

# Define the list of commands to generate
commands = [
    "adelante",
    "atrás",
    "izquierda",
    "derecha",
    "alto",
    "ir",
    "sí",
    "no",
    "uno",
    "dos",
    "tres",
    "cuatro",
    "cinco"
]

NUM_VARIATIONS = 10
output_dir = "synthetic_audio_data"
# Clear directory if needed, or just overwrite
os.makedirs(output_dir, exist_ok=True)

def speed_change(sound, speed=1.0):
    # Manually override the frame_rate. This increases the pitch and the speed.
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

print(f"Generating synthetic audio for {len(commands)} commands (x{NUM_VARIATIONS} variations) into '{output_dir}'...")

for i, command in enumerate(commands):
    # 2. Generate variations
    for j in range(NUM_VARIATIONS):
        try:
            # Randomly select an accent (TLD) to simulate different voices
            # 'com.mx': Mexico, 'es': Spain, 'us': USA (often distinct)
            selected_tld = random.choice(['com.mx', 'es', 'us'])
            
            # Generate base speech from text using gTTS with selected accent
            tts = gTTS(text=command, lang='es', tld=selected_tld, slow=False)
            mp3_fp = BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            
            base_audio = AudioSegment.from_file(mp3_fp, format="mp3")

            # Vary speed between 0.8 (slower/deeper) and 1.3 (faster/higher)
            speed = random.uniform(0.8, 1.3)
            
            aug_audio = speed_change(base_audio, speed)

            # Save
            # Filename format: command_index_variation.wav
            wav_file_path = os.path.join(output_dir, f"{command.replace(' ', '_')}_{i}_{j}.wav")
            
            aug_audio.export(wav_file_path, format="wav")
        except Exception as e:
             print(f"Error generating variation {j} for '{command}': {e}")
            
    print(f"Generated {NUM_VARIATIONS} variations for: {command}")

print("Audio generation complete.")