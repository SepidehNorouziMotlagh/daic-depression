# src/make_kaggle_manifest.py
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT   = Path(__file__).resolve().parent.parent
MANIFEST  = PROJECT / "data" / "manifests"
LOCAL_AUD = PROJECT / "data" / "participant_audio"

DATASET_SLUG = "daic-woz-cleaned-audio"
KAGGLE_AUDIO_DIR = f"/kaggle/input/{DATASET_SLUG}"


def _normalize_columns(df):
    df = df.copy()
    df.columns = [
        str(c).strip().lstrip("\ufeff").lower().replace(" ", "_")
        for c in df.columns
    ]
    return df


def _find_col(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None


def norm(df, split):
    df = _normalize_columns(df)

    pid_col = _find_col(df, "participant_id")
    if pid_col is None:
        raise KeyError(f"{split}: no participant_id column; got {list(df.columns)}")
    df = df.rename(columns={pid_col: "participant_id"})
    df["participant_id"] = df["participant_id"].astype(int).astype(str)
    df["split"] = split

    # Binary label — column name differs between train/dev and test
    bin_col   = _find_col(df, "phq8_binary", "phq_binary")
    score_col = _find_col(df, "phq8_score",  "phq_score")

    if bin_col is not None:
        df["label"] = df[bin_col].astype(int)
    elif score_col is not None:
        df["label"] = (df[score_col].astype(float) >= 10).astype(int)
        print(f"[{split}] no binary column; derived label from {score_col} >= 10")
    else:
        raise KeyError(f"{split}: no label columns; got {list(df.columns)}")

    df["PHQ8_Score"] = df[score_col] if score_col else np.nan
    df["audio_path"] = KAGGLE_AUDIO_DIR + "/" + df["participant_id"] + ".wav"
    return df


def main():
    train = norm(pd.read_csv(MANIFEST / "train_split_Depression_AVEC2017.csv"), "train")
    dev   = norm(pd.read_csv(MANIFEST / "dev_split_Depression_AVEC2017.csv"),   "dev")
    test  = norm(pd.read_csv(MANIFEST / "full_test_split.csv"),                  "test")

    full = pd.concat([train, dev, test], ignore_index=True)

    available = {p.stem for p in LOCAL_AUD.glob("*.wav")}
    manifest = full[full["participant_id"].isin(available)].reset_index(drop=True)
    manifest = manifest[["participant_id", "PHQ8_Score", "label", "split", "audio_path"]]

    out_path = MANIFEST / "daic_manifest.csv"
    manifest.to_csv(out_path, index=False)

    print(f"Wrote {out_path}")
    print(f"Total subjects: {len(manifest)}")
    print("\nPer split:")
    print(manifest.groupby("split")["label"].agg(["count", "sum", "mean"]))
    print("\nFirst 3 rows:")
    print(manifest.head(3).to_string())

    missing = full[~full["participant_id"].isin(available)]["participant_id"].tolist()
    if missing:
        print(f"\n⚠️  Dropped {len(missing)} PIDs with no local .wav: {missing}")


if __name__ == "__main__":
    main()