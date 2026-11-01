# Audio-Only Depression Detection on DAIC-WOZ

Fine-tuning of **Wav2Vec 2.0** and **Whisper** for binary depression detection on
the DAIC-WOZ dataset (AVEC 2017 distribution). This repository contains
implementation code and aggregate experimental results only.

## Dataset Notice

This project uses the Distress Analysis Interview Corpus (DAIC-WOZ), provided
by the University of Southern California Institute for Creative Technologies.

The dataset is **not included in this repository**. Access must be obtained
directly from the official providers, subject to their End-User License
Agreement. Users are responsible for obtaining authorization before accessing
or using the dataset.

This repository contains implementation code and research documentation only.
It does not grant permission to access, redistribute, or commercially use
DAIC-WOZ.

If you reference or build on this work, cite both the corpus (LREC 2014) and the
AVEC 2017 challenge (see *Citation* below).

## What This Repository Contains

- `src/` — preprocessing pipeline: transcript-guided participant speech
  extraction (removing interviewer turns, silence, and PII-scrubbed segments),
  manifest construction, and download utility.
- `training/` — nine fine-tuning scripts spanning full fine-tuning, partial
  fine-tuning, frozen-feature probes, layer summation, time masking, and LoRA
  adaptation across both required models.
- `results/results_summary.json` — aggregate metrics for all nine runs.
- `docs/` — technical report.

## What This Repository Does **Not** Contain

- DAIC-WOZ audio, transcripts, feature files, or CSVs of any kind.
- Dataset manifests or train/dev/test splits.
- Model checkpoints or trained weights.
- Per-participant predictions or evaluation outputs.

## Results at a Glance

All numbers on the **official AVEC 2017 test split** (47 subjects: 14 depressed,
33 non-depressed, verified against `full_test_split.csv`).

| Run | Model | Configuration | Test AUC | Test F1 |
|---|---|---|---|---|
| 1 | Wav2Vec2-base | Full FT + attention pooling + class weights | 0.543 | **0.320** |
| 2 | Wav2Vec2-base | Top-4 unfreeze + mean pool + dev-threshold tuning | 0.483 | 0.256 |
| 3 | Wav2Vec2-base | Frozen features + aggregated linear probe | 0.526 | **0.400** |
| **4** | **Wav2Vec2-base** | **Frozen encoder + small MLP head** | **0.578** | 0.000 |
| 5 | Wav2Vec2-base | 7 s non-overlap + layer summation + time masking | 0.509 | 0.000 |
| 6 | Wav2Vec2-base | LoRA (r=8) on attention Q, V | 0.517 | 0.000 |
| 7 | Whisper-small | Frozen feature probe | 0.483 | 0.000 |
| 8 | Whisper-small | Partial FT (last 4 encoder layers) | 0.472 | 0.000 |
| **9** | **Whisper-small** | **LoRA (r=8) on attention Q, V** | **0.541** | 0.000 |

**Best Wav2Vec2 AUC:** 0.578 (run 4, frozen encoder).
**Best Wav2Vec2 F1:** 0.400 (run 3, aggregated probe).
**Best Whisper AUC:** 0.541 (run 9, LoRA).

Runs 4–9 report F1 = 0.000 at threshold 0.5 with the identical confusion matrix
`[[33, 0], [14, 0]]` — every subject predicted "not depressed." This is not a
broken implementation; it is the class-prior miscalibration signature of a
weakly-informative classifier trained on 28%-positive data. Runs 1–3 produce
nonzero F1 by explicitly pushing prediction probabilities upward (class
weighting in runs 1 and 3; dev-threshold tuning in run 2), at the cost of
lower AUC. **AUC is the primary metric for this class distribution.**

## Preprocessing Overview

Raw DAIC-WOZ audio is approximately 90% interviewer speech, silence, and
PII-redacted spans. The preprocessing pipeline (`src/preprocess.py`) extracts
only participant speech using the transcript's ground-truth speaker and
timestamp columns:

1. Parse `{pid}_TRANSCRIPT.csv` (tab-separated).
2. Remove rows whose `value` contains `scrubbed_entry` (PII-redaction markers
   that correspond to silent waveform spans).
3. Retain only participant turns (drop rows where `speaker` contains "Ellie").
4. Discard segments shorter than 1.0 s (backchannels, breath noises).
5. Slice audio by transcript timestamps with ±50 ms pad on each end; discard
   chunks whose sample-energy indicates near-total silence.
6. Concatenate the retained chunks into a single 16 kHz mono WAV per
   participant.

Result: ~2.6 GB of cleaned audio across 188 usable participants, with
2–10 minutes of participant speech per subject.

## Experimental Design

- **Splits:** official AVEC 2017 train / dev / test, no re-splitting.
- **Windowing:** 10 s windows, 5 s stride (run 5 uses 7 s / 7 s).
- **Batch:** 2 physical × 8 accumulation = 16 effective.
- **Optimizer:** AdamW with weight decay 0.01.
- **Schedule:** cosine with 10% linear warmup.
- **Precision:** AMP (fp16) on a Tesla T4 (Colab free tier).
- **Reproducibility:** single seed (`torch.manual_seed(42)`); note that runs are
  not bit-exact across machines due to CUDA nondeterminism.

Individual configurations and hyperparameters are documented in each script's
module docstring.

## Reproducing

You must obtain DAIC-WOZ yourself; it is not included here.

1. Request access from USC-ICT (see link in the dataset provider's website)
   and sign the EULA.
2. Copy `.env.example` to `.env` and fill in `USC_BASE_URL` with the base
   download URL from your approval email.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install peft        # only needed for runs 6 and 9
Run the preprocessing pipeline (streaming, low disk):

bash
python src/pipeline.py
Adapt and run any training script in training/; each reads
DAIC_AUDIO_DIR, DAIC_CKPT_DIR, and DAIC_RESULTS_DIR from the
environment (see training/README.md).

Reproducibility Notes
The test split contains 14 positive subjects (PIDs 308, 309, 311, 332,
337, 354, 359, 365, 384, 405, 410, 421, 453, 461) and 33 negative. Some
AVEC 2017 companion documentation cites a count of 15; this appears to
reflect a different labeling or a pre-release count and does not match the
distributed full_test_split.csv. The released file is authoritative.

Session 440 was downloadable but returned a corrupt archive
(File is not a zip file after matching Content-Length). Session 440 is
in the dev split; test evaluation is unaffected.

Training was run on the Colab free tier (T4 GPU). Session disconnects were
handled by saving optimizer + scheduler + model state to Google Drive after
every epoch and auto-resuming from the last completed epoch.

Citation
If you build on this work, cite both the corpus and the AVEC 2017 challenge:

bibtex
@inproceedings{gratch2014distress,
  title     = {The Distress Analysis Interview Corpus of human and computer interviews},
  author    = {Gratch, Jonathan and Artstein, Ron and Lucas, Gale and Stratou, Giota
               and Scherer, Stefan and Nazarian, Angela and Wood, Rachel
               and Boberg, Jill and DeVault, David and Marsella, Stacy
               and Traum, David and Rizzo, Skip and Morency, Louis-Philippe},
  booktitle = {Proceedings of the Language Resources and Evaluation Conference (LREC)},
  year      = {2014}
}

@inproceedings{ringeval2017avec,
  title     = {AVEC 2017: Real-life Depression, and Affect Recognition Workshop and Challenge},
  author    = {Ringeval, Fabien and Schuller, Bj{\"o}rn and Valstar, Michel and others},
  booktitle = {Proceedings of the 7th Annual Workshop on Audio/Visual Emotion Challenge},
  year      = {2017}
}
License
Code in this repository is provided for academic reference only. It does not
grant any rights to the DAIC-WOZ dataset, which remains subject to its own
End-User License Agreement.