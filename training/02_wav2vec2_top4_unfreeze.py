"""
Run 2 — Diagnostic run: unfreeze last 4 transformer layers, mean pooling,
per-subject window cap, no class weighting, threshold tuned on dev.

Epoch 1 hits dev AUC 0.688 (the highest dev AUC in the whole project), but the
model re-enters the memorisation regime by epoch 3. The dev-tuned threshold
(0.25) overfits the 12-positive dev split and does not transfer to test.

Result (test, at dev-tuned threshold 0.25):
    Test AUC = 0.483
    Test F1  = 0.256
    Test Acc = 0.383
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

SR, WIN_SEC, STRIDE_SEC = 16000, 10, 5
BATCH_SIZE, GRAD_ACCUM  = 2, 8
LR, EPOCHS              = 3e-5, 12
MAX_WIN_PER_SUBJECT     = 8
MODEL_NAME              = "facebook/wav2vec2-base"
DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu"

AUDIO_DIR   = os.environ.get("DAIC_AUDIO_DIR", "./data/participant_audio")
MANIFEST    = os.path.join(AUDIO_DIR, "daic_manifest.csv")
CKPT_DIR    = os.path.join(os.environ.get("DAIC_CKPT_DIR", "./checkpoints"), "run02_top4")
RESULTS_DIR = os.environ.get("DAIC_RESULTS_DIR", "./results")
os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
torch.manual_seed(42); np.random.seed(42); random.seed(42)


class DAICWindowDataset(Dataset):
    """Resamples a random subset of up to max_per_subject windows per subject each epoch."""

    def __init__(self, df, processor, max_per_subject=None):
        self.df = df.reset_index(drop=True)
        self.proc = processor
        self.max_per_subject = max_per_subject
        self.all_windows = {}
        for i, row in self.df.iterrows():
            n = sf.info(row["audio_path"]).frames
            nwin = max(1, (n - SR * WIN_SEC) // (SR * STRIDE_SEC) + 1)
            self.all_windows[i] = list(range(nwin))
        self.index = []
        self._resample()

    def _resample(self):
        self.index = []
        for i, wins in self.all_windows.items():
            chosen = random.sample(wins, self.max_per_subject) \
                if self.max_per_subject and len(wins) > self.max_per_subject else wins
            for w in chosen:
                self.index.append((i, w))

    def resample(self):
        if self.max_per_subject:
            self._resample()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, w = self.index[k]
        row = self.df.iloc[i]
        s = w * SR * STRIDE_SEC
        chunk, _ = sf.read(row["audio_path"], start=s, frames=SR*WIN_SEC, dtype="float32")
        if len(chunk) < SR*WIN_SEC:
            chunk = np.pad(chunk, (0, SR*WIN_SEC - len(chunk)))
        inputs = self.proc(chunk, sampling_rate=SR, return_tensors="pt")
        return {
            "input_values": inputs.input_values.squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.long),
            "subject_idx": torch.tensor(i, dtype=torch.long),
        }


class Wav2Vec2Top4(nn.Module):
    """Freeze all encoder layers except the last 4; mean pool; linear classify."""

    def __init__(self, model_name=MODEL_NAME, num_classes=2, dropout=0.2):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        self.encoder.feature_extractor._freeze_parameters()
        n_layers = len(self.encoder.encoder.layers)
        for i, layer in enumerate(self.encoder.encoder.layers):
            if i < n_layers - 4:
                for p in layer.parameters():
                    p.requires_grad = False
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(h, num_classes)

    def forward(self, input_values):
        out = self.encoder(input_values).last_hidden_state
        return self.classifier(self.dropout(out.mean(dim=1)))


def evaluate_subject_level(model, loader, df, threshold=0.5):
    model.eval()
    probs_by_subject = {}
    with torch.no_grad():
        for batch in loader:
            x = batch["input_values"].to(DEVICE)
            probs = F.softmax(model(x), dim=-1)[:, 1].cpu().numpy()
            for p, si in zip(probs, batch["subject_idx"].numpy()):
                probs_by_subject.setdefault(int(si), []).append(float(p))
    idxs = sorted(probs_by_subject)
    y_true = np.array([df.iloc[i]["label"] for i in idxs])
    y_prob = np.array([np.mean(probs_by_subject[i]) for i in idxs])
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1":  f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
        "cm":  confusion_matrix(y_true, y_pred).tolist(),
        "y_true": y_true, "y_prob": y_prob,
    }


def find_best_threshold(y_true, y_prob, thresholds=np.arange(0.1, 0.91, 0.05)):
    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        f1 = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = thr, f1
    return best_thr, best_f1


def main():
    df = pd.read_csv(MANIFEST)
    df["audio_path"] = df["audio_path"].apply(lambda p: os.path.join(AUDIO_DIR, os.path.basename(p)))
    df = df[df["audio_path"].apply(os.path.exists)].reset_index(drop=True)
    train_df = df[df.split == "train"].reset_index(drop=True)
    dev_df   = df[df.split == "dev"].reset_index(drop=True)
    test_df  = df[df.split == "test"].reset_index(drop=True)
    print(f"Subjects: train={len(train_df)} dev={len(dev_df)} test={len(test_df)}")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    train_ds = DAICWindowDataset(train_df, processor, max_per_subject=MAX_WIN_PER_SUBJECT)
    dev_ds   = DAICWindowDataset(dev_df,   processor)
    test_ds  = DAICWindowDataset(test_df,  processor)
    print(f"Windows per epoch: train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = Wav2Vec2Top4().to(DEVICE)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

    enc_params  = [p for p in model.encoder.parameters()  if p.requires_grad]
    head_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": LR}, {"params": head_params, "lr": LR * 10}],
        weight_decay=0.01,
    )
    steps_per_epoch = len(train_loader) // GRAD_ACCUM
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(0.1 * steps_per_epoch * EPOCHS), steps_per_epoch * EPOCHS
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    best_f1, best_state, best_thr = -1.0, None, 0.5
    for epoch in range(1, EPOCHS + 1):
        train_ds.resample()
        model.train(); running = 0.0
        optimizer.zero_grad(); t0 = time.time()
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch:02d}", leave=False)):
            x = batch["input_values"].to(DEVICE, non_blocking=True)
            y = batch["label"].to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                loss = F.cross_entropy(model(x), y) / GRAD_ACCUM
            scaler.scale(loss).backward()
            running += loss.item() * GRAD_ACCUM
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                scheduler.step(); optimizer.zero_grad()

        dev = evaluate_subject_level(model, dev_loader, dev_df)
        thr, thr_f1 = find_best_threshold(dev["y_true"], dev["y_prob"])
        print(f"epoch {epoch:02d} | loss={running/len(train_loader):.4f} "
              f"| dev AUC={dev['auc']:.3f} F1@0.5={dev['f1']:.3f} F1@{thr:.2f}={thr_f1:.3f} "
              f"| {time.time()-t0:.0f}s")

        if thr_f1 > best_f1:
            best_f1, best_thr = thr_f1, thr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(CKPT_DIR, "wav2vec2_best.pt"))
            print(f"  ↳ new best (dev F1={thr_f1:.3f} @ thr={thr:.2f})")

    print(f"\n=== Final test evaluation (best dev checkpoint, thr={best_thr:.2f}) ===")
    model.load_state_dict(best_state)
    test = evaluate_subject_level(model, test_loader, test_df, threshold=best_thr)
    print(f"TEST  F1={test['f1']:.3f}  Acc={test['acc']:.3f}  AUC={test['auc']:.3f}")
    print(f"Confusion matrix:\n{np.array(test['cm'])}")

    with open(os.path.join(RESULTS_DIR, "results_run02.json"), "w") as f:
        json.dump({"best_dev_f1": best_f1, "best_thr": float(best_thr),
                   **{f"test_{k}": v for k, v in test.items() if k != "y_true" and k != "y_prob"}},
                  f, indent=2)


if __name__ == "__main__":
    main()