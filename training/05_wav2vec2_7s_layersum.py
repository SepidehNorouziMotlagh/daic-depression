"""
Run A — 7s non-overlapping windows + layer summation + time masking.

Three simultaneous changes:
  1. 7s windows with 7s stride (non-overlapping) → no near-duplicates
  2. Sum over all 12 transformer layers + LayerNorm → combine acoustic + semantic
  3. SpecAugment-style time masking → effective data augmentation
Also evaluates both mean and confidence-weighted aggregation.

Diagnosis: dev AUC rose steadily (0.494 → 0.557) but test AUC stayed random
(0.509). The three changes collectively hurt vs. run 4's frozen baseline.

Result (test, best dev checkpoint):
    Test AUC (mean agg)  = 0.509, F1 = 0.000
    Test AUC (attn agg)  = 0.506, F1 = 0.000
    Confusion matrix = [[33, 0], [14, 0]]  (both aggregation methods)
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

SR              = 16000
WIN_SEC         = 7
STRIDE_SEC      = 7
BATCH_SIZE      = 2
GRAD_ACCUM      = 8
LR              = 1e-5
EPOCHS          = 5
MAX_WIN_PER_SUBJECT = 16
MODEL_NAME      = "facebook/wav2vec2-base"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

AUDIO_DIR   = os.environ.get("DAIC_AUDIO_DIR", "./data/participant_audio")
MANIFEST    = os.path.join(AUDIO_DIR, "daic_manifest.csv")
CKPT_DIR    = os.path.join(os.environ.get("DAIC_CKPT_DIR", "./checkpoints"), "run05_7s_layersum")
RESULTS_DIR = os.environ.get("DAIC_RESULTS_DIR", "./results")
os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
torch.manual_seed(42); np.random.seed(42); random.seed(42)


def time_mask(chunk, sr=SR, num_masks=2, max_mask_sec=1.0, p=0.5):
    """SpecAugment-style time masking: randomly zero out 1-2 spans with prob p."""
    if random.random() > p:
        return chunk
    chunk = chunk.copy()
    max_span = int(max_mask_sec * sr)
    min_span = max(sr // 4, 1)
    for _ in range(num_masks):
        span = random.randint(min_span, max_span)
        if span >= len(chunk):
            continue
        start = random.randint(0, len(chunk) - span)
        chunk[start:start + span] = 0.0
    return chunk


class DAICWindowDataset(Dataset):
    def __init__(self, df, processor, max_per_subject=None, augment=False):
        self.df = df.reset_index(drop=True)
        self.proc = processor
        self.max_per_subject = max_per_subject
        self.augment = augment
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
        if self.augment:
            chunk = time_mask(chunk)
        inputs = self.proc(chunk, sampling_rate=SR, return_tensors="pt")
        return {
            "input_values": inputs.input_values.squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.long),
            "subject_idx": torch.tensor(i, dtype=torch.long),
        }


class Wav2Vec2LayerSum(nn.Module):
    """
    Frozen encoder with layer-summation head:
    - Sum transformer layers 1..12 (skip embedding layer 0)
    - LayerNorm to normalise the summed magnitudes
    - Per-frame 768→256 projection
    - Mean pool over time
    - Dropout + linear classify
    """

    def __init__(self, model_name=MODEL_NAME, num_classes=2, hidden=256, dropout=0.3):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False
        h = self.encoder.config.hidden_size
        self.layer_norm = nn.LayerNorm(h)
        self.projector  = nn.Linear(h, hidden)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_values):
        with torch.no_grad():
            outputs = self.encoder(input_values, output_hidden_states=True)
            stacked = torch.stack(outputs.hidden_states[1:], dim=0)
            summed  = stacked.sum(dim=0)
        x = self.layer_norm(summed)
        x = self.projector(x)
        x = x.mean(dim=1)
        return self.classifier(self.dropout(x))


def aggregate_window_probs(probs, method="mean", temperature=0.2):
    """mean: simple average. attention: confidence-weighted (|p-0.5| sharpened by softmax)."""
    probs = np.asarray(probs, dtype=np.float64)
    if method == "mean":
        return probs.mean()
    conf = np.abs(probs - 0.5) / max(temperature, 1e-8)
    conf = np.exp(conf - conf.max())
    weights = conf / conf.sum()
    return float((weights * probs).sum())


def evaluate_subject_level(model, loader, df, threshold=0.5, agg_method="mean"):
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
    y_prob = np.array([aggregate_window_probs(probs_by_subject[i], method=agg_method) for i in idxs])
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1":  f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
        "cm":  confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    df = pd.read_csv(MANIFEST)
    df["audio_path"] = df["audio_path"].apply(lambda p: os.path.join(AUDIO_DIR, os.path.basename(p)))
    df = df[df["audio_path"].apply(os.path.exists)].reset_index(drop=True)
    train_df = df[df.split == "train"].reset_index(drop=True)
    dev_df   = df[df.split == "dev"].reset_index(drop=True)
    test_df  = df[df.split == "test"].reset_index(drop=True)
    print(f"Subjects: train={len(train_df)} dev={len(dev_df)} test={len(test_df)}")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    train_ds = DAICWindowDataset(train_df, processor, max_per_subject=MAX_WIN_PER_SUBJECT, augment=True)
    dev_ds   = DAICWindowDataset(dev_df,   processor)
    test_ds  = DAICWindowDataset(test_df,  processor)
    print(f"Windows per epoch: train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = Wav2Vec2LayerSum().to(DEVICE)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01,
    )
    steps_per_epoch = len(train_loader) // GRAD_ACCUM
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(0.1 * steps_per_epoch * EPOCHS), steps_per_epoch * EPOCHS,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    best_auc, best_state = -1.0, None
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

        dev_mean = evaluate_subject_level(model, dev_loader, dev_df, agg_method="mean")
        dev_attn = evaluate_subject_level(model, dev_loader, dev_df, agg_method="attention")
        print(f"epoch {epoch:02d} | loss={running/len(train_loader):.4f} "
              f"| dev mean AUC={dev_mean['auc']:.3f} | dev attn AUC={dev_attn['auc']:.3f} "
              f"| {time.time()-t0:.0f}s")

        if dev_attn["auc"] > best_auc:
            best_auc = dev_attn["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(CKPT_DIR, "wav2vec2_runA_best.pt"))
            print(f"  ↳ new best (dev attn AUC={best_auc:.3f})")

    print(f"\n=== Final test evaluation (best dev checkpoint) ===")
    model.load_state_dict(best_state)
    test_mean = evaluate_subject_level(model, test_loader, test_df, agg_method="mean")
    test_attn = evaluate_subject_level(model, test_loader, test_df, agg_method="attention")
    print(f"Test (mean agg)  F1={test_mean['f1']:.3f}  Acc={test_mean['acc']:.3f}  AUC={test_mean['auc']:.3f}")
    print(f"Test (attn agg)  F1={test_attn['f1']:.3f}  Acc={test_attn['acc']:.3f}  AUC={test_attn['auc']:.3f}")

    with open(os.path.join(RESULTS_DIR, "results_run05.json"), "w") as f:
        json.dump({"best_dev_auc_attn": best_auc,
                   "test_mean_agg": test_mean, "test_attn_agg": test_attn}, f, indent=2)


if __name__ == "__main__":
    main()