"""
Run 9 — Whisper-small LoRA fine-tuning (BEST Whisper configuration).

Wraps WhisperForAudioClassification with LoRA adapters (r=8, alpha=16) on the
attention q_proj and v_proj layers. Base model weights frozen; only LoRA
matrices (~295K) and classifier head (~197K) trainable.

Diagnosis: LoRA helped Whisper (test AUC 0.541 vs 0.483 frozen / 0.472 partial),
but dev AUC peaked at epoch 1 (0.538) and declined — overfitting started
immediately. The dev AUC at peak is still below Wav2Vec2's frozen baseline.

Result (test, threshold 0.5):
    Test AUC = 0.541
    Test F1  = 0.000
    Test Acc = 0.702
    Confusion matrix = [[33, 0], [14, 0]]
"""
import os, random, time, json
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperForAudioClassification, WhisperProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

SR, WIN_SEC, STRIDE_SEC = 16000, 10, 5
BATCH_SIZE, GRAD_ACCUM  = 2, 8
EPOCHS                  = 5
MAX_WIN_PER_SUBJECT     = 4
LORA_R, LORA_ALPHA      = 8, 16
LORA_DROPOUT            = 0.05
LORA_TARGETS            = ["q_proj", "v_proj"]
LR_LORA, LR_HEAD        = 2e-4, 1e-4
MODEL_NAME              = "openai/whisper-small"
DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu"

AUDIO_DIR   = os.environ.get("DAIC_AUDIO_DIR", "./data/participant_audio")
MANIFEST    = os.path.join(AUDIO_DIR, "daic_manifest.csv")
CKPT_DIR    = os.path.join(os.environ.get("DAIC_CKPT_DIR", "./checkpoints"), "run09_whisper_lora")
RESULTS_DIR = os.environ.get("DAIC_RESULTS_DIR", "./results")
os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
torch.manual_seed(42); np.random.seed(42); random.seed(42)


class DAICWhisperDataset(Dataset):
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
            "input_features": inputs.input_features.squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.long),
            "subject_idx": torch.tensor(i, dtype=torch.long),
        }


def build_whisper_lora(model_name, num_classes=2):
    """Wrap WhisperForAudioClassification with LoRA on attention projections."""
    model = WhisperForAudioClassification.from_pretrained(model_name, num_labels=num_classes)
    for p in model.parameters():
        p.requires_grad = False
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT, bias="none",
    ))
    for p in model.classifier.parameters():
        p.requires_grad = True
    if hasattr(model, "projector"):
        for p in model.projector.parameters():
            p.requires_grad = True
    return model


def evaluate_subject_level(model, loader, df, threshold=0.5):
    model.eval()
    probs_by_subject = {}
    with torch.no_grad():
        for batch in loader:
            x = batch["input_features"].to(DEVICE)
            probs = F.softmax(model(x).logits, dim=-1)[:, 1].cpu().numpy()
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
    }


def main():
    df = pd.read_csv(MANIFEST)
    df["audio_path"] = df["audio_path"].apply(lambda p: os.path.join(AUDIO_DIR, os.path.basename(p)))
    df = df[df["audio_path"].apply(os.path.exists)].reset_index(drop=True)
    train_df = df[df.split == "train"].reset_index(drop=True)
    dev_df   = df[df.split == "dev"].reset_index(drop=True)
    test_df  = df[df.split == "test"].reset_index(drop=True)
    print(f"Subjects: train={len(train_df)} dev={len(dev_df)} test={len(test_df)}")

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    train_ds = DAICWhisperDataset(train_df, processor, max_per_subject=MAX_WIN_PER_SUBJECT)
    dev_ds   = DAICWhisperDataset(dev_df,   processor)
    test_ds  = DAICWhisperDataset(test_df,  processor)
    print(f"Windows per epoch: train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_whisper_lora(MODEL_NAME).to(DEVICE)
    model.print_trainable_parameters()

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" not in n]
    print(f"LoRA: {sum(p.numel() for p in lora_params)/1e6:.3f}M | "
          f"Head: {sum(p.numel() for p in head_params)/1e6:.3f}M")

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LR_LORA},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=0.01)
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
            x = batch["input_features"].to(DEVICE, non_blocking=True)
            y = batch["label"].to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                loss = F.cross_entropy(model(x).logits, y) / GRAD_ACCUM
            scaler.scale(loss).backward()
            running += loss.item() * GRAD_ACCUM
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                scheduler.step(); optimizer.zero_grad()

        dev = evaluate_subject_level(model, dev_loader, dev_df)
        print(f"epoch {epoch:02d} | loss={running/len(train_loader):.4f} "
              f"| dev AUC={dev['auc']:.3f} F1={dev['f1']:.3f} | {time.time()-t0:.0f}s")
        if dev["auc"] > best_auc:
            best_auc = dev["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(CKPT_DIR, "whisper_lora_best.pt"))
            print(f"  ↳ new best (dev AUC={best_auc:.3f})")

    print(f"\n=== Final test evaluation (best dev AUC={best_auc:.3f}) ===")
    model.load_state_dict(best_state)
    test = evaluate_subject_level(model, test_loader, test_df)
    print(f"TEST  AUC={test['auc']:.3f}  F1={test['f1']:.3f}  Acc={test['acc']:.3f}")
    print(f"Confusion matrix:\n{np.array(test['cm'])}")

    with open(os.path.join(RESULTS_DIR, "results_run09.json"), "w") as f:
        json.dump({"best_dev_auc": best_auc, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                   **{f"test_{k}": v for k, v in test.items()}}, f, indent=2)


if __name__ == "__main__":
    main()