This directory contains the implementation of all nine experimental configurations evaluated for audio-only depression detection on the DAIC-WOZ dataset. Each script is standalone and can be run from PyCharm, a Colab cell, or the command line.

Quick Reference
File	Run	Model	Config	Test AUC	Test F1
01_wav2vec2_full_finetune.py	1	Wav2Vec2-base	Full FT + attention + class weights	0.543	0.320
02_wav2vec2_top4_unfreeze.py	2	Wav2Vec2-base	Top-4 unfreeze + mean pool + thr tune	0.483	0.256
03_wav2vec2_linear_probe.py	3	Wav2Vec2-base	Frozen features + aggregated linear probe	0.526	0.400
04_wav2vec2_frozen_head.py	4	Wav2Vec2-base	Frozen encoder + small head	0.578	0.000
05_wav2vec2_7s_layersum.py	5	Wav2Vec2-base	7s non-overlap + layer sum + time mask	0.509	0.000
06_wav2vec2_lora.py	6	Wav2Vec2-base	LoRA r=8 on attention Q,V	0.517	0.000
07_whisper_frozen_probe.py	7	Whisper-small	Frozen features + linear probe	0.483	0.000
08_whisper_partial_ft.py	8	Whisper-small	Partial FT (last 4 encoder layers)	0.472	0.000
09_whisper_lora.py	9	Whisper-small	LoRA r=8 on attention Q,V	0.541	0.000
Best Wav2Vec2 AUC: Run 4 (frozen encoder) — 0.578
Best Wav2Vec2 F1: Run 3 (aggregated probe) — 0.400
Best Whisper AUC: Run 9 (LoRA) — 0.541

All numbers are on the official AVEC 2017 test split (47 subjects: 14 positive, 33 negative), verified against full_test_split.csv.

Setup
Each script reads three environment variables. Set them before running.

Variable	Default	Purpose
DAIC_AUDIO_DIR	./data/participant_audio	Cleaned participant WAVs + daic_manifest.csv
DAIC_CKPT_DIR	./checkpoints	Checkpoints (each script appends its own subfolder)
DAIC_RESULTS_DIR	./results	Per-run results_runNN.json files
Example — PyCharm / PowerShell
powershell
$env:DAIC_AUDIO_DIR   = "F:\sepide\Projects\daic_depression\data\participant_audio"
$env:DAIC_CKPT_DIR    = "F:\sepide\Projects\daic_depression\checkpoints"
$env:DAIC_RESULTS_DIR = "F:\sepide\Projects\daic_depression\results"

python training\04_wav2vec2_frozen_head.py
Example — Colab
python
import os
os.environ["DAIC_AUDIO_DIR"]   = "/content/drive/MyDrive/daic_depression/participant_audio"
os.environ["DAIC_CKPT_DIR"]    = "/content/drive/MyDrive/daic_depression/checkpoints"
os.environ["DAIC_RESULTS_DIR"] = "/content/drive/MyDrive/daic_depression/results"

!python training/04_wav2vec2_frozen_head.py
Prerequisites
You must obtain DAIC-WOZ yourself (the dataset is not included in this repository) and run the preprocessing pipeline in ../src/ to produce data/participant_audio/*.wav and data/manifests/daic_manifest.csv.

Runtime dependencies are in ../requirements.txt, plus peft for runs 6 and 9:

bash
pip install -r requirements.txt
pip install peft
What Each Run Tests
Run 1 — Full fine-tune + attention pooling + class weights.
Trains the entire Wav2Vec2 transformer with a learnable attention-pooling head, class-weighted loss (pos_weight=2.57), and label smoothing. Tests whether end-to-end fine-tuning with a moderate loss compensates for the small training set. Outcome: catastrophic overfitting — train loss collapses from 0.52 → 0.17 in 4 epochs (the perfect-separation floor for the weighted+smoothed loss), while dev AUC stays below 0.50. The model memorises participant voice-identity.

Run 2 — Top-4 unfreeze + window cap + threshold tuning.
Freezes all but the last 4 transformer layers, uses mean pooling, caps each subject to 8 windows per epoch (fresh random sample each epoch), removes class weighting, and selects a threshold on dev. Tests whether reducing capacity and rebalancing loss fixes run 1's problem. Outcome: epoch 1 hits the highest dev AUC in the project (0.688), but the model re-enters the memorisation regime by epoch 3. The dev-tuned threshold (0.25) overfits the 12-positive dev split and does not transfer to test.

Run 3 — Frozen features + aggregated linear probe.
Freezes the encoder, mean-pools 768-D features per window, aggregates to subject-level (mean across ~65 windows), and trains a small MLP with class weighting. Tests whether frozen features already separate classes. Outcome: 0.17 AUC gap between dev (0.696) and test (0.526) — the features look promising on dev but do not generalise. Notably, F1 = 0.400 here because class weights in the head push positive probabilities above 0.5 for some subjects.

Run 4 — Fully frozen encoder + small MLP head (BEST Wav2Vec2).
Trains a 200K-parameter head on 4 windows per subject with no class weights, no label smoothing, gentle LR (1e-5), 3 epochs. Tests the hypothesis that if the frozen features carry weak signal, the smallest possible trainable unit is the right one. Outcome: 0.578 test AUC — the best in the project. All probabilities sit below 0.5 at threshold 0.5 (class prior 28%, no weighting), so F1 = 0. AUC is the informative metric for this run.

Run 5 (Run A) — 7s windows + layer summation + time masking.
Replaces 10s/5s overlapping windows with 7s non-overlapping (eliminating near-duplicates), sums the last 12 transformer layers + LayerNorm (instead of last hidden state only), applies SpecAugment-style time masking during training, and evaluates both mean and confidence-weighted aggregation. Tests three commonly-cited literature changes at once. Outcome: dev AUC rises (0.494 → 0.557) but test AUC collapses to 0.509. The three changes hurt jointly — layer sum + LayerNorm normalised away the weak signal; time masking injected noise; and 16 windows/subject re-opened the memorisation shortcut.

Run 6 — Wav2Vec2 LoRA (r=8).
Wraps Wav2Vec2-base with LoRA adapters on attention q_proj/v_proj (r=8, alpha=16, ~295K trainable) plus a small head (~197K). This actually fine-tunes the encoder through the adapter path — the middle ground between full FT (overfits) and fully frozen (limited signal). Outcome: train loss drops, dev AUC rises monotonically (0.470 → 0.557), but test AUC is 0.517 — below the frozen baseline. Adding trainable parameters to the encoder hurt rather than helped.

Run 7 — Whisper frozen feature probe.
Same strategy as run 3 but for Whisper-small: extract mean-pooled frozen features, aggregate to subject-level, train a small MLP with class weighting. Tests whether Whisper's ASR-optimised representations carry depression signal. Outcome: test AUC 0.483 — below the random baseline. Whisper's frozen features do not carry usable depression signal even when the head overfits dev (dev AUC 0.573 → test 0.483).

Run 8 — Whisper partial fine-tune (last 4 layers).
Unfreezes the last 4 of 12 encoder layers (~28.4M trainable), uses a two-LR setup (encoder at 5e-6, head at 1e-4), 6 epochs. Tests whether adaptation can create signal where the frozen probe found none. Outcome: dev AUC stays below 0.50 from epoch 1 and decreases over 6 epochs (0.486 → 0.466). Train loss barely moves (0.638 → 0.579). This is anti-learning — partial fine-tuning cannot adapt Whisper's representations on 107 subjects.

Run 9 — Whisper LoRA (r=8, BEST Whisper).
Wraps WhisperForAudioClassification with LoRA on attention q_proj/v_proj (~295K trainable) plus the classifier head (~197K). Parameter-efficient fine-tuning for Whisper. Outcome: test AUC 0.541 — the best Whisper result, +0.058 over the frozen probe and +0.069 over partial FT. But dev AUC peaks at epoch 1 (0.538) and declines, indicating immediate overfitting. Still below Wav2Vec2's frozen baseline (0.578).

Interpreting the F1 = 0.000 Results
Six of the nine runs report F1 = 0.000 at threshold 0.5, with the identical confusion matrix [[33, 0], [14, 0]] — every subject predicted "not depressed." This is not a bug and not a broken implementation. It is the canonical signature of a weakly-informative classifier trained on imbalanced data (28% positive):

The model learns to output probabilities near the class prior for both classes.

Depressed subjects score slightly higher than non-depressed (hence AUC > 0.5), but neither group crosses 0.5.

Runs 1, 2, 3 avoid F1 = 0 by explicitly pushing probabilities up (class weights in runs 1, 3; threshold tuning in run 2), at the cost of lower AUC.

For a fair summary of model quality, report AUC as the primary metric. F1 at threshold 0.5 is misleading for imbalanced data.

What the Runs Jointly Establish
Within the tested configurations at 107 training subjects, increasing trainable parameters beyond ~200K does not improve test AUC and often degrades it. Runs 4 and 3 (both ~200K trainable) produce the two best Wav2Vec2 results; runs 1, 2, and 6 (all with 500K+ or 33M+ trainable) all score worse on test AUC. This is an observation about the tested configurations, not a universal claim about dataset size — windowing, pooling, architecture, label noise, and participant variability likely also contribute and were not fully isolated.

The most defensible framing for the project's conclusion: audio-only single-model fine-tuning on DAIC-WOZ is constrained at ~0.55–0.58 AUC in our tested configurations, and literature's stronger results (F1 0.75–0.85) require multi-modal fusion (audio + text + video) or hand-crafted acoustic feature augmentation, both outside the scope of this task.

Output Files
Every run writes a single results_runNN.json to DAIC_RESULTS_DIR with the final metrics. Best checkpoints are saved to DAIC_CKPT_DIR/runNN_*/. The exact JSON schema differs slightly between runs (see each script's final json.dump block), but all include best_dev_*, test_auc, test_f1, test_acc, and test_cm.

See the project's main README.md for the dataset notice, preprocessing overview, and EULA-compliance statement.