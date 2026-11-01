# Audio-Only Depression Detection on DAIC-WOZ

An experimental study of **Wav2Vec 2.0** and **Whisper** for binary depression detection using audio from the **Distress Analysis Interview Corpus (DAIC-WOZ)**, distributed as part of the AVEC 2017 challenge.

This repository contains the implementation, experimental configurations, aggregate evaluation results, and technical documentation. The DAIC-WOZ dataset and all dataset-derived files are excluded.

## Dataset and Usage Notice

This project uses the Distress Analysis Interview Corpus (DAIC-WOZ), provided by the University of Southern California Institute for Creative Technologies (USC-ICT).

The dataset is **not included in this repository**. Access must be requested directly from the official providers and is subject to their End-User License Agreement (EULA).

In particular:

- DAIC-WOZ is made available for **non-commercial research purposes only**.
- The dataset, or any part of it, must not be redistributed.
- Users must obtain their own authorization before accessing or using the data.
- This repository does not grant access to the dataset or permission to redistribute or commercially use it.

This repository is intended to share implementation code and research findings, not the underlying dataset. Please review and comply with the dataset's licensing terms before reproducing the experiments.

## Project Overview

The objective was to investigate whether pretrained speech representation models can support binary depression detection from participant speech.

Two pretrained models were evaluated:

- **Wav2Vec 2.0 Base**
- **Whisper Small**

Nine experimental configurations were implemented, covering full and partial fine-tuning, frozen-feature classification, layer aggregation, temporal augmentation, and LoRA adaptation.

The experiments use the official AVEC 2017 train, development, and test partitions. The prediction target is binary depression status based on the PHQ-8 threshold.

## Repository Contents

| Directory / File | Description |
|---|---|
| `src/` | Audio preprocessing, participant-speech extraction, manifest construction, and download utilities |
| `training/` | Nine training scripts covering the experimental configurations |
| `results/results_summary.json` | Aggregate evaluation metrics for all nine runs |
| `docs/` | Technical report and project documentation |
| `requirements.txt` | Python dependencies |
| `.env.example` | Example environment configuration |

The training scripts document their individual model configurations and hyperparameters.

## What Is Not Included

To respect the DAIC-WOZ EULA, this repository does not contain:

- DAIC-WOZ audio recordings.
- Transcripts or interview text.
- Original dataset CSV files or split manifests.
- Extracted audio features or cached representations.
- Model checkpoints or trained weights.
- Participant-level predictions or evaluation outputs.

Only implementation code, documentation, and aggregate experimental results are included.

## Experimental Results

The following table summarizes performance on the official AVEC 2017 test partition, comprising **47 participants: 14 with depression labels and 33 without depression labels**.

| Run | Model | Configuration | Test AUC | Test F1 |
|---|---|---|---:|---:|
| 1 | Wav2Vec 2.0 Base | Full fine-tuning + attention pooling + class weights | 0.543 | 0.320 |
| 2 | Wav2Vec 2.0 Base | Top-4 layer unfreezing + mean pooling + development-set threshold tuning | 0.483 | 0.256 |
| 3 | Wav2Vec 2.0 Base | Frozen features + aggregated linear probe | 0.526 | **0.400** |
| **4** | **Wav2Vec 2.0 Base** | **Frozen encoder + lightweight MLP classification head** | **0.578** | 0.000 |
| 5 | Wav2Vec 2.0 Base | 7-second windows + layer summation + time masking | 0.509 | 0.000 |
| 6 | Wav2Vec 2.0 Base | LoRA (r=8) on attention query and value projections | 0.517 | 0.000 |
| 7 | Whisper Small | Frozen-feature classification probe | 0.483 | 0.000 |
| 8 | Whisper Small | Partial fine-tuning of the last four encoder layers | 0.472 | 0.000 |
| **9** | **Whisper Small** | **LoRA (r=8) on attention query and value projections** | **0.541** | 0.000 |

**Best results by model and metric:**

- **Best overall test AUC:** 0.578 — Wav2Vec 2.0, frozen encoder (Run 4).
- **Best overall test F1:** 0.400 — Wav2Vec 2.0, aggregated linear probe (Run 3).
- **Best Whisper test AUC:** 0.541 — Whisper LoRA (Run 9).

### Interpreting the Results

Runs 4–9 produced an F1 score of 0.000 at the default classification threshold of 0.5. All six runs produced the same test confusion matrix:

`[[33, 0], [14, 0]]`

In other words, these configurations classified every test participant as not depressed.

Runs 1–3 produced nonzero F1 scores under different training or threshold configurations. However, their results also illustrate the difficulty of obtaining useful classification performance from this relatively small, imbalanced, subject-level dataset.

The highest observed AUC was 0.578, indicating only weak ranking performance on the test set. The results should therefore be interpreted as exploratory experimental findings, not evidence of clinically useful depression detection.

AUC provides a threshold-independent view of ranking performance, while F1 reflects performance at a particular classification threshold. Both are reported because they capture different aspects of model behavior.

## Audio Preprocessing

The preprocessing pipeline extracts participant speech from the original interview recordings using the accompanying transcript timestamps and speaker annotations.

The pipeline is implemented in `src/preprocess.py` and performs the following operations:

1. Parses the participant transcript file.
2. Removes transcript entries marked as `scrubbed_entry`, corresponding to redacted portions.
3. Retains participant speech and excludes interviewer turns.
4. Discards segments shorter than 1.0 second.
5. Extracts audio segments using transcript timestamps, with a 50 ms padding interval on each side.
6. Filters segments with near-zero audio energy.
7. Concatenates the retained segments into a single 16 kHz mono audio file per participant.

The resulting processed collection contained approximately **2.6 GB of audio across 188 usable participants**, with approximately 2–10 minutes of retained participant speech per subject.

The preprocessing pipeline operates on the authorized local dataset files. Those files are not included in this repository.

## Experimental Setup

| Component | Configuration |
|---|---|
| Dataset partitions | Official AVEC 2017 train, development, and test splits |
| Target | Binary depression classification |
| Wav2Vec 2.0 model | Base |
| Whisper model | Small |
| Standard windowing | 10-second windows with a 5-second stride |
| Run 5 windowing | 7-second non-overlapping windows |
| Physical batch size | 2 |
| Gradient accumulation | 8 steps |
| Effective batch size | 16 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Learning-rate schedule | Cosine decay with 10% linear warmup |
| Precision | Automatic mixed precision (FP16) |
| Hardware | NVIDIA Tesla T4 GPU |
| Random seed | 42 |

Experiments were conducted using Google Colab's free-tier GPU environment.

Training scripts save model, optimizer, and scheduler states to support recovery after interrupted sessions. Runs use a fixed random seed, but exact bit-for-bit reproducibility is not guaranteed across different hardware and CUDA environments.

## Reproducing the Experiments

### 1. Obtain dataset access

Request access to DAIC-WOZ through the official provider:

[DAIC-WOZ — USC Institute for Creative Technologies](https://dcapswoz.ict.usc.edu/)

You must obtain authorization and sign the applicable EULA before downloading or using the dataset.

### 2. Configure the environment

Clone the repository and create a local environment configuration:

```bash
git clone <repository-url>
cd <repository-directory>

cp .env.example .env
```

Configure the required environment variables in `.env`, including the authorized download URL if you intend to use the download utility.

**Do not commit `.env` or any credentials, access tokens, or private download URLs.**

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For the LoRA experiments (Runs 6 and 9), install PEFT:

```bash
pip install peft
```

### 4. Prepare the audio

After obtaining the dataset and configuring the required local paths, run:

```bash
python src/pipeline.py
```

The preprocessing pipeline prepares the participant audio for the training scripts.

### 5. Run an experiment

Configure the local data, checkpoint, and results directories using the environment variables:

- `DAIC_AUDIO_DIR`
- `DAIC_CKPT_DIR`
- `DAIC_RESULTS_DIR`

Then execute the desired training script from `training/`.

See `training/README.md` and the individual script documentation for experiment-specific instructions.

## Reproducibility Notes and Limitations

- The official test partition contains 47 participants: 14 positive and 33 negative according to the released evaluation labels.
- One development participant session (Session 440) could not be processed because its downloaded archive was corrupt. It was therefore excluded from the usable development data. The official test partition was unaffected.
- Training was performed on a Tesla T4 GPU through Google Colab.
- Checkpoints were saved between epochs to support recovery from session interruptions.
- The experiments use a relatively small subject-level dataset. Results may be sensitive to participant variability, model configuration, windowing, pooling strategy, and classification threshold.
- The experiments do not establish clinical validity or generalization to other datasets and populations.

The accompanying technical report provides additional details about the experimental configurations, implementation decisions, observed results, and limitations.

## Citation

If you publish research using DAIC-WOZ, acknowledge the corpus using the following citation:

**DAIC-WOZ**

```bibtex
@inproceedings{gratch2014distress,
  title     = {The Distress Analysis Interview Corpus of Human and Computer Interviews},
  author    = {Gratch, Jonathan and Artstein, Ron and Lucas, Gale and Stratou, Giota
               and Scherer, Stefan and Nazarian, Angela and Wood, Rachel
               and Boberg, Jill and DeVault, David and Marsella, Stacy
               and Traum, David and Rizzo, Skip and Morency, Louis-Philippe},
  booktitle = {Proceedings of the Language Resources and Evaluation Conference (LREC)},
  year      = {2014}
}
```

For work specifically related to the AVEC 2017 challenge, also cite:

**AVEC 2017**

```bibtex
@inproceedings{ringeval2017avec,
  title     = {AVEC 2017: Real-life Depression, and Affect Recognition Workshop and Challenge},
  author    = {Ringeval, Fabien and Schuller, Björn and Valstar, Michel and others},
  booktitle = {Proceedings of the 7th Annual Workshop on Audio/Visual Emotion Challenge},
  year      = {2017}
}
```

## Code and Dataset Terms

The code in this repository is shared for research and educational reference. The DAIC-WOZ dataset is governed independently by its provider's EULA.

Nothing in this repository grants rights to access, redistribute, or commercially use DAIC-WOZ. Users are responsible for obtaining their own dataset authorization and complying with the applicable terms.
