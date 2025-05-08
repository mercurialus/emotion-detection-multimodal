import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Use raw strings for Windows-style paths
audio_root = r"C:\Users\Harshil\Documents\Codes\emotion-detection-multimodal\data\ravdess-emotional-speech-audio\versions\1"
mfcc_root = r"C:\Users\Harshil\Documents\Codes\emotion-detection-multimodal\processed\mfcc"
spec_root = r"C:\Users\Harshil\Documents\Codes\emotion-detection-multimodal\processed\spectrogram"

# Emotion ID to label mapping
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Extract emotion from filename
def extract_emotion(filename):
    # RAVDESS format: XX-XX-EMOTION-XX-XX-XX-XX.wav
    emotion_id = filename.split('-')[2]
    return emotion_map.get(emotion_id, 'unknown')

# Save MFCC or Spectrogram image
def save_image(matrix, save_path):
    plt.figure(figsize=(3, 3))
    plt.axis('off')
    librosa.display.specshow(matrix)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Process a single file
def process_file(filepath):
    print(f"Processing: {filepath}")
    try:
        y, sr = librosa.load(filepath, sr=None)
        filename = os.path.basename(filepath)
        emotion = extract_emotion(filename)

        # Create directories
        mfcc_dir = os.path.join(mfcc_root, emotion)
        spec_dir = os.path.join(spec_root, emotion)
        os.makedirs(mfcc_dir, exist_ok=True)
        os.makedirs(spec_dir, exist_ok=True)

        # Extract and save MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_path = os.path.join(mfcc_dir, filename.replace('.wav', '.png'))
        save_image(mfcc, mfcc_path)

        # Extract and save Mel Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        spec_path = os.path.join(spec_dir, filename.replace('.wav', '.png'))
        save_image(S_dB, spec_path)

    except Exception as e:
        print(f"Failed to process {filepath}: {e}")

# Walk through all .wav files
for root, _, files in os.walk(audio_root):
    for file in files:
        if file.endswith(".wav"):
            full_path = os.path.join(root, file)
            process_file(full_path)

print("Processing complete.")