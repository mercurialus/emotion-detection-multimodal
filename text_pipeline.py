import csv,json,os,re,torch
from pathlib import Path
from tqdm import tqdm
from faster_whisper import WhisperModel

root = Path(r"C:\Users\Harshil\Documents\Codes\emotion-detection-multimodal\data\ravdess-emotional-speech-audio\versions\1")
out_csv = Path(r"C:\Users\Harshil\Documents\Codes\emotion-detection-multimodal\data\ravdess-emotional-speech-audio\processed\transcripts.csv")
MODEL_SIZE= "base"

idx_to_emotion = [          
    "neutral","calm","happy","sad",
    "angry","fearful","disgust","surprised"
]

def emotion_from_path(p:Path)->str:
    idx = int(p.stem.split("-")[2])-1
    return idx_to_emotion[idx]

whisper = WhisperModel(MODEL_SIZE, device="cuda" if torch.cuda.is_available() else "cpu",compute_type="float32")

rows = []

for wav in tqdm(root.rglob("*.wav")):
    segments, _ = whisper.transcribe(str(wav), beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments)
    text = re.sub(r"[^A-Za-z0-9' ]+", " ", text).lower().strip()
    if len(text) < 3:
        text = f"i am very {emotion_from_path(wav)}"
    rows.append((str(wav), text, emotion_from_path(wav)))

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["wav_path","transcript","label"])
    csv.writer(f).writerows(rows)
print(f"Wrote {len(rows)} rows to {out_csv}")