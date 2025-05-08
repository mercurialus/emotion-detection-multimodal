import os, csv, re, tqdm, random, numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import efficientnet_b0
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 32
EPOCHS = 25
LR_HEAD = 3e-4
LR_FINETUNE = 3e-5
PATIENCE = 4
LABELS = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]
label2idx = {l:i for i,l in enumerate(LABELS)}

PATH_AUDIO_MFCC = Path(r"C:\Users\Harshil\Documents\Codes\emotion-detection-multimodal\processed\mfcc")
PATH_CSV_TRANS = Path(r"C:\Users\Harshil\Documents\Codes\emotion-detection-multimodal\data\ravdess-emotional-speech-audio\processed\transcripts.csv")
CKPT_CNN = Path("efficientnet_b0_best_model.pth")
CKPT_RNN = Path("best_text_rnn.pt")

TOK = re.compile(r"[A-Za-z']+")

def tokenize(s):
    return TOK.findall(s.lower())

def build_vocab(rows, min_freq=2):
    from collections import Counter
    freqs = Counter(tok for txt,_ in rows for tok in tokenize(txt))
    vocab = {"<pad>":0, "<unk>":1}
    for w,c in freqs.items():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab

def numericalise(txt, vocab, max_len=20):
    ids = [vocab.get(t,1) for t in tokenize(txt)][:max_len]
    return ids + [0]*(max_len-len(ids))

def load_mfcc_image(path):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return tfm(img)

class RavdessMulti(Dataset):
    def __init__(self, csv_path, mfcc_root, vocab, max_len=20):
        rows = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                wav = Path(r["wav_path"])
                label = r["label"]
                mfcc = mfcc_root / label / wav.with_suffix(".png").name
                rows.append((mfcc, numericalise(r["transcript"], vocab, max_len), label2idx[label]))
        self.rows = rows
        self.vocab = vocab
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        img_path, ids, lab = self.rows[idx]
        img = load_mfcc_image(img_path)
        return img, torch.tensor(ids, dtype=torch.long), lab

def collate(batch):
    imgs, seqs, labs = zip(*batch)
    imgs = torch.stack(imgs)
    labs = torch.tensor(labs)
    lens = torch.tensor([len(s) for s in seqs])
    seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    return imgs, seqs, lens, labs

def build_cnn():
    m = efficientnet_b0(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, len(LABELS))
    return m

class EmotionRNN(nn.Module):
    def __init__(self, n_vocab, emb_dim=50, hid=128, layers=1, dr=0.3):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hid, layers, batch_first=True, bidirectional=True, dropout=dr if layers>1 else 0.0)
        self.fc = nn.Sequential(nn.Dropout(dr), nn.Linear(hid*2, len(LABELS)))
    def forward(self,x):
        x = self.emb(x)
        _,(h,_) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(h)

class AudioBackbone(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.feats = nn.Sequential(net.features, net.avgpool, nn.Flatten())
        self.out_dim = 1280
        for p in self.parameters():
            p.requires_grad = False
    def forward(self,x):
        return self.feats(x)

class TextBackbone(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.emb, self.lstm = net.emb, net.lstm
        self.out_dim = net.lstm.hidden_size*2
        for p in self.parameters():
            p.requires_grad = False
    def forward(self,x,lens):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        _,(h,_) = self.lstm(packed)
        return torch.cat([h[-2], h[-1]], dim=1)

class FusionHead(nn.Module):
    def __init__(self, a_dim, t_dim, n_cls=8, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm1d(a_dim+t_dim), nn.Linear(a_dim+t_dim, 256), nn.ReLU(), nn.Dropout(p), nn.Linear(256, n_cls))
    def forward(self,z):
        return self.net(z)

class MultiModal(nn.Module):
    def __init__(self, a_back, t_back):
        super().__init__()
        self.a = a_back
        self.t = t_back
        self.head = FusionHead(a_back.out_dim, t_back.out_dim, len(LABELS))
    def forward(self,img,txt,lens):
        z = torch.cat([self.a(img), self.t(txt,lens)],1)
        return self.head(z)

def run_epoch(model, dataloader, train=True, optimizer=None):
    model.train() if train else model.eval()
    losses, preds, golds = [], [], []
    for imgs, txts, lens, y in tqdm.tqdm(dataloader, leave=False):
        imgs, txts, lens, y = imgs.to(DEVICE), txts.to(DEVICE), lens.to(DEVICE), y.to(DEVICE)
        with torch.set_grad_enabled(train):
            out = model(imgs, txts, lens)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        preds.extend(out.argmax(1).cpu().tolist())
        golds.extend(y.cpu().tolist())
    acc = accuracy_score(golds, preds)
    return np.mean(losses), acc

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()

    with open(PATH_CSV_TRANS, newline='', encoding='utf-8') as f:
        rows_csv = [(r["transcript"], r["label"]) for r in csv.DictReader(f)]
    vocab = build_vocab(rows_csv)

    full_ds = RavdessMulti(PATH_CSV_TRANS, PATH_AUDIO_MFCC, vocab)
    N_val = int(0.2 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [len(full_ds)-N_val, N_val], generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, collate_fn=collate, num_workers=2, pin_memory=True)

    audio_cnn = build_cnn().to(DEVICE)
    audio_cnn.load_state_dict(torch.load(CKPT_CNN, map_location=DEVICE))
    text_rnn = EmotionRNN(len(vocab)).to(DEVICE)
    text_rnn.load_state_dict(torch.load(CKPT_RNN, map_location=DEVICE))

    audio_back = AudioBackbone(audio_cnn)
    text_back = TextBackbone(text_rnn)
    model = MultiModal(audio_back, text_back).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optim_head = torch.optim.Adam(model.head.parameters(), lr=LR_HEAD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_head, mode='max', factor=0.5, patience=2)

    train_losses, val_losses = [], []
    best_acc, impatience = 0, 0
    for epoch in range(1, EPOCHS+1):
        tloss, tacc = run_epoch(model, train_dl, train=True, optimizer=optim_head)
        vloss, vacc = run_epoch(model, val_dl, train=False)
        scheduler.step(vacc)
        train_losses.append(tloss)
        val_losses.append(vloss)
        print(f"[{epoch:02d}] train-loss {tloss:.4f} | val-loss {vloss:.4f} | val-acc {vacc*100:.2f}%)")
        if vacc > best_acc:
            best_acc, impatience = vacc, 0
            torch.save(model.state_dict(), "best_fusion.pt")
        else:
            impatience += 1
            if impatience >= PATIENCE:
                print("Early stop.")
                break

    print("Best fusion val-accuracy:", best_acc*100)

    for p in model.a.parameters():
        p.requires_grad = True
    for p in model.t.parameters():
        p.requires_grad = True

    optim_all = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)
    for epoch in range(1, 6):
        model.train()
        losses = []
        for imgs, txts, lens, y in tqdm.tqdm(train_dl, leave=False):
            imgs, txts, lens, y = imgs.to(DEVICE), txts.to(DEVICE), lens.to(DEVICE), y.to(DEVICE)
            out = model(imgs, txts, lens)
            loss = criterion(out, y)
            optim_all.zero_grad()
            loss.backward()
            optim_all.step()
            losses.append(loss.item())
        print(f"finetune-epoch {epoch} | loss {np.mean(losses):.4f}")

    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for imgs, txts, lens, y in val_dl:
            p = model(imgs.to(DEVICE), txts.to(DEVICE), lens.to(DEVICE)).argmax(1)
            all_p.extend(p.cpu().tolist())
            all_y.extend(y.tolist())
    print(classification_report(all_y, all_p, target_names=LABELS))

    cm = confusion_matrix(all_y, all_p)
    disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    plt.figure(figsize=(8,6))
    plt.plot(train_losses, marker='o', label='Train Loss')
    plt.plot(val_losses, marker='x', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve.png')
    plt.show()
