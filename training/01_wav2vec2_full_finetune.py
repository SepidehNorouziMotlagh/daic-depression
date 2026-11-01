"""
Run 1 — Full encoder fine-tuning + attention pooling + class weights.

Trains the entire Wav2Vec2-base transformer with a learnable attention-pooling
head, class-weighted cross-entropy (pos_weight = n_neg/n_pos ≈ 2.57), and label
smoothing. No window cap: every epoch sees the full ~8700 training windows.

Diagnosis: catastrophic overfitting. Train loss collapses 0.52 → 0.17 within
4 epochs (the perfect-separation floor under the weighted + smoothed loss),
while dev AUC stays below 0.50 — the model memorises participant identity.

Result (official AVEC 2017 test split, 47 subjects, 14 positive):
    Test AUC = 0.543
    Test F1  = 0.320   (nonzero only because class weights push positive probabilities up)
    Test Acc = 0.638
    Confusion matrix = [[26, 7], [10, 4]]
"""
import os, random, time, json
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

# --------------------------- Config ---------------------------
SR          = 16000
WIN_SEC     = 10
STRIDE_SEC  = 5
BATCH_SIZE  = 2
GRAD_ACCUM  = 8
LR          = 1e-5
EPOCHS      = 12
MODEL_NAME  = "facebook/wav2vec2-base"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

AUDIO_DIR   = os.environ.get("DAIC_AUDIO_DIR", "./data/participant_audio")
MANIFEST    = os.path.join(AUDIO_DIR, "daic_manifest.csv")
CKPT_DIR    = os.path.join(os.environ.get("DAIC_CKPT_DIR", "./checkpoints"), "run01_full_ft")
RESULTS_DIR = os.environ.get("DAIC_RESULTS_DIR", "./results")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(42); np.random.seed(42); random.seed(42)
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# --------------------------- Dataset --------------------------
class DAICWindowDataset(Dataset):
    """One example per 10s window; subject_idx kept for subject-level aggregation."""

    def __init__(self, df, processor):
        self.df = df.reset_index(drop=True)
        self.proc = processor
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
        chunk, _ = sf.read(row["audio_path"], start=s, frames=SR * WIN_SEC, dtype="float32")
        if len(chunk) < SR * WIN_SEC:
            chunk = np.pad(chunk, (0, SR * WIN_SEC - len(chunk)))
        inputs = self.proc(chunk, sampling_rate=SR, return_tensors="pt")
        return {
            "input_values": inputs.input_values.squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.long),
            "subject_idx": torch.tensor(i, dtype=torch.long),
        }


# --------------------------- Model ----------------------------
class Wav2Vec2Depression(nn.Module):
    """Encoder + additive attention pooling + linear classifier."""

    def __init__(self, model_name=MODEL_NAME, num_classes=2, dropout=0.2):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        self.encoder.feature_extractor._freeze_parameters()
        h = self.encoder.config.hidden_size
        self.attn = nn.Sequential(nn.Linear(h, 128), nn.Tanh(), nn.Linear(128, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(h, num_classes)

    def forward(self, input_values):
        out = self.encoder(input_values).last_hidden_state     # (B, T, H)
        w = torch.softmax(self.attn(out), dim=1)               # (B, T, 1)
        pooled = (w * out).sum(dim=1)                          # (B, H)
        return self.classifier(self.dropout(pooled))


# --------------------------- Eval -----------------------------
def evaluate_subject_level(model, loader, df, threshold=0.5):
    """Aggregate window-level probabilities per subject (mean), then score."""
    model.eval()
    subject_probs = {}
    with torch.no_grad():
        for batch in loader:
            x = batch["input_values"].to(DEVICE)
            probs = F.softmax(model(x), dim=-1)[:, 1].cpu().numpy()
            for p, si in zip(probs, batch["subject_idx"].numpy()):
                subject_probs.setdefault(int(si), []).append(float(p))
    idxs = sorted(subject_probs)
    y_true = np.array([df.iloc[i]["label"] for i in idxs])
    y_prob = np.array([np.mean(subject_probs[i]) for i in idxs])
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1":  f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
        "cm":  confusion_matrix(y_true, y_pred).tolist(),
    }


# --------------------------- Main -----------------------------
def main():
    df = pd.read_csv(MANIFEST)
    df["audio_path"] = df["audio_path"].apply(
        lambda p: os.path.join(AUDIO_DIR, os.path.basename(p))
    )
    df = df[df["audio_path"].apply(os.path.exists)].reset_index(drop=True)

    train_df = df[df.split == "train"].reset_index(drop=True)
    dev_df   = df[df.split == "dev"].reset_index(drop=True)
    test_df  = df[df.split == "test"].reset_index(drop=True)
    print(f"Subjects: train={len(train_df)} dev={len(dev_df)} test={len(test_df)}")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    train_ds  = DAICWindowDataset(train_df, processor)
    dev_ds    = DAICWindowDataset(dev_df,   processor)
    test_ds   = DAICWindowDataset(test_df,  processor)
    print(f"Windows: train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = Wav2Vec2Depression().to(DEVICE)
    n_pos = int(train_df["label"].sum()); n_neg = len(train_df) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    print(f"Class balance — pos={n_pos} neg={n_neg} pos_weight={pos_weight:.2f}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    steps_per_epoch = len(train_loader) // GRAD_ACCUM
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(0.1 * steps_per_epoch * EPOCHS), steps_per_epoch * EPOCHS
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    weight_tensor = torch.tensor([1.0, pos_weight], device=DEVICE)

    best_f1, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train(); running = 0.0
        optimizer.zero_grad(); t0 = time.time()
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch:02d}", leave=False)):
            x = batch["input_values"].to(DEVICE, non_blocking=True)
            y = batch["label"].to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                logits = model(x)
                loss = F.cross_entropy(logits, y, weight=weight_tensor,
                                       label_smoothing=0.05) / GRAD_ACCUM
            scaler.scale(loss).backward()
            running += loss.item() * GRAD_ACCUM
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                scheduler.step(); optimizer.zero_grad()

        dev = evaluate_subject_level(model, dev_loader, dev_df)
        print(f"epoch {epoch:02d} | loss={running/len(train_loader):.4f} "
              f"| dev F1={dev['f1']:.3f} acc={dev['acc']:.3f} auc={dev['auc']:.3f} "
              f"| {time.time()-t0:.0f}s")

        if dev["f1"] > best_f1:
            best_f1 = dev["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(CKPT_DIR, "wav2vec2_best.pt"))
            print(f"  ↳ new best (dev F1={best_f1:.3f}) saved")

    print("\n=== Final test evaluation (best dev checkpoint) ===")
    model.load_state_dict(best_state)
    test = evaluate_subject_level(model, test_loader, test_df)
    print(f"TEST  F1={test['f1']:.3f}  Acc={test['acc']:.3f}  AUC={test['auc']:.3f}")
    print(f"Confusion matrix:\n{np.array(test['cm'])}")

    with open(os.path.join(RESULTS_DIR, "results_run01.json"), "w") as f:
        json.dump({"best_dev_f1": best_f1, **{f"test_{k}": v for k, v in test.items()}}, f, indent=2)


if __name__ == "__main__":
    main()