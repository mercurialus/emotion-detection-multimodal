import os, re, csv, tqdm
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = Path(r"C:\Users\Harshil\Documents\Codes\emotion-detection-multimodal"
                r"\data\ravdess-emotional-speech-audio\processed\transcripts.csv")

LABELS = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]
label2idx = {l:i for i,l in enumerate(LABELS)}

def read_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for r in rdr: rows.append((r["transcript"], label2idx[r["label"]]))
    return rows

data = read_csv(CSV_PATH)

TOK = re.compile(r"[A-Za-z']+")
def tokenize(s): return TOK.findall(s.lower())

all_tokens = [tok for sent,_ in data for tok in tokenize(sent)]
freqs = Counter(all_tokens)

vocab = {"<pad>":0, "<unk>":1}
for w,c in freqs.items():
    if c >= 2:
        vocab[w] = len(vocab)

def numericalise(sent, max_len=20):
    ids = [vocab.get(t,1) for t in tokenize(sent)]
    ids = ids[:max_len]
    return ids + [0]*(max_len-len(ids))

class TranscriptDS(Dataset):
    def __init__(self, rows):
        self.sent = [numericalise(s) for s,_ in rows]
        self.lab  = [l for _,l in rows]
    def __len__(self): return len(self.lab)
    def __getitem__(self,i):
        return torch.tensor(self.sent[i], dtype=torch.long), self.lab[i]

FULL_DS = TranscriptDS(data)

labels = np.array([l for _,l in data])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(np.zeros(len(labels)), labels))

train_ds = torch.utils.data.Subset(FULL_DS, train_idx)
val_ds   = torch.utils.data.Subset(FULL_DS, val_idx)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False)

EMB_DIM = 50
class EmotionRNN(nn.Module):
    def __init__(self, n_vocab, emb_dim=50, hid=128, layers=1, dr=0.3):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, emb_dim, padding_idx=0)
        nn.init.xavier_normal_(self.emb.weight)
        self.lstm = nn.LSTM(emb_dim, hid, layers,
                            batch_first=True, bidirectional=True, dropout=dr)
        self.fc = nn.Sequential(
            nn.Dropout(dr),
            nn.Linear(hid*2, len(LABELS))
        )
    def forward(self,x):
        x = self.emb(x)
        _,(h,_) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(h)

model = EmotionRNN(len(vocab)).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
crit  = nn.CrossEntropyLoss()

def evaluate(dl):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x,y in dl:
            x,y = x.to(DEVICE),y.to(DEVICE)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += len(y)
    return correct / total

best_acc, epochs_no_improve = 0, 0
EPOCHS, PATIENCE = 15, 3
train_losses, val_accuracies = [], []

val_losses = []
print("Len of vocab:", len(vocab))
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    loop = tqdm.tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}")
    for x, y in loop:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        epoch_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = epoch_loss / len(train_dl)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = crit(out, y)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_dl)
    val_losses.append(avg_val_loss)

    val_acc = evaluate(val_dl)
    val_accuracies.append(val_acc)
    print(f"Val Acc: {val_acc * 100:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_text_rnn.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("Early stopping.")
            break

print(f"Best validation accuracy: {best_acc*100:.2f}%")


plt.figure(figsize=(8,6))
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Val Loss', marker='x')
plt.title("Train vs Validation Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("train_vs_val_loss.png")
plt.show()
