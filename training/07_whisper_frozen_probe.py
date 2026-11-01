"""
Run 7 — Whisper-small frozen feature probe.

Same diagnostic as Run 3 but for Whisper: extract mean-pooled features from
the frozen encoder, aggregate to subject-level, train a small MLP. Class
weights used during head training; selection by dev AUC.

Diagnosis: test AUC 0.483 — below the random baseline. Whisper's ASR-optimised
representations do not carry depression-relevant signal in a form a lightweight
head can extract, even when the head overfits dev (dev AUC 0.573 → test 0.483).

Result (test, threshold 0.5):
    Test AUC = 0.483
    Test F1  = 0.000
    Test Acc = 0.702
    Confusion matrix = [[33, 0], [14, 0]]
"""
import os, random, json
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperModel, WhisperProcessor
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

SR, WIN_SEC, STRIDE_SEC = 16000, 10, 5
MODEL_NAME    = "openai/whisper-small"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS        = 50
LR            = 1e-3
WEIGHT_DECAY  = 0.1
DROPOUT       = 0.3

AUDIO_DIR   = os.environ.get("DAIC_AUDIO_DIR", "./data/participant_audio")
MANIFEST    = os.path.join(AUDIO_DIR, "daic_manifest.csv")
RESULTS_DIR = os.environ.get("DAIC_RESULTS_DIR", "./results")
os.makedirs(RESULTS_DIR, exist_ok=True)
torch.manual_seed(42); np.random.seed(42); random.seed(42)


class WindowFeatureDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.index = []
        for i, row in self.df.iterrows():
            n = sf.info(row["audio_path"]).frames
            nwin = max(1, (n - SR * WIN_SEC) // (SR * STRIDE_SEC) + 1)
            for w in range(nwin):
                self.index.append((i, w))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, w = self.index[k]
        row = self.df.iloc[i]
        s = w * SR * STRIDE_SEC
        chunk, _ = sf.read(row["audio_path"], start=s, frames=SR*WIN_SEC, dtype="float32")
        if len(chunk) < SR*WIN_SEC:
            chunk = np.pad(chunk, (0, SR*WIN_SEC - len(chunk)))
        return torch.from_numpy(chunk), i


def extract_features(df, encoder, proc):
    ds = WindowFeatureDataset(df)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
    feats, subjs = [], []
    encoder.eval()
    with torch.no_grad():
        for chunk, si in tqdm(dl, desc="extracting", leave=False):
            x = proc(list(chunk.numpy()), sampling_rate=SR,
                     return_tensors="pt").input_features.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                h = encoder(x).last_hidden_state.mean(dim=1)
            feats.append(h.cpu()); subjs.append(si)
    return torch.cat(feats), torch.cat(subjs)


def main():
    df = pd.read_csv(MANIFEST)
    df["audio_path"] = df["audio_path"].apply(lambda p: os.path.join(AUDIO_DIR, os.path.basename(p)))
    df = df[df["audio_path"].apply(os.path.exists)].reset_index(drop=True)

    encoder = WhisperModel.from_pretrained(MODEL_NAME).encoder.to(DEVICE).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    proc = WhisperProcessor.from_pretrained(MODEL_NAME)

    splits = {}
    for split in ["train", "dev", "test"]:
        sub_df = df[df.split == split].reset_index(drop=True)
        print(f"Extracting {split}...")
        X, s = extract_features(sub_df, encoder, proc)
        uniq = sorted(s.unique().tolist())
        X_agg = torch.stack([X[s == u].mean(0) for u in uniq]).to(DEVICE)
        y = torch.tensor(sub_df["label"].values, dtype=torch.long, device=DEVICE)
        splits[split] = (X_agg, y)

    Xtr, ytr = splits["train"]
    Xdv, ydv = splits["dev"]
    Xte, yte = splits["test"]

    head = nn.Sequential(
        nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(128, 2),
    ).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_pos = int((ytr == 1).sum()); n_neg = int((ytr == 0).sum())
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, n_neg / max(n_pos, 1)], device=DEVICE))

    best_auc, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        head.train(); opt.zero_grad()
        loss_fn(head(Xtr), ytr).backward(); opt.step()
        head.eval()
        with torch.no_grad():
            p_dv = F.softmax(head(Xdv), -1)[:, 1].cpu().numpy()
        auc = roc_auc_score(ydv.cpu().numpy(), p_dv)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state); head.eval()
    with torch.no_grad():
        p_te = F.softmax(head(Xte), -1)[:, 1].cpu().numpy()
    y_true = yte.cpu().numpy()
    y_pred = (p_te >= 0.5).astype(int)

    result = {
        "dev_auc":  float(best_auc),
        "test_auc": float(roc_auc_score(y_true, p_te)),
        "test_f1":  float(f1_score(y_true, y_pred, zero_division=0)),
        "test_acc": float(accuracy_score(y_true, y_pred)),
        "test_cm":  confusion_matrix(y_true, y_pred).tolist(),
    }
    print(f"\n=== Whisper FROZEN PROBE ===")
    print(f"Dev  AUC={result['dev_auc']:.3f}")
    print(f"Test AUC={result['test_auc']:.3f}  F1={result['test_f1']:.3f}  Acc={result['test_acc']:.3f}")
    print(f"Confusion matrix:\n{np.array(result['test_cm'])}")

    with open(os.path.join(RESULTS_DIR, "results_run07.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()