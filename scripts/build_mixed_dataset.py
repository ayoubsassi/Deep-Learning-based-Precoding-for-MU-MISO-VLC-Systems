import pickle
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = [
    ("2_users_data_19.2k.pkl", 2, 16000),
    ("old_data_4users_40k.pkl", 4, 16000),
    ("6_users_data.pkl", 6, 16000),
    ("3_users_data.pkl", 3, 16000),
    ("5_users_data.pkl", 5, 16000),
    ("1_users_data.pkl", 1, 16000),

]

M = 6  # transmitters
K_MAX = 6
OUTPUT_PATH = Path("mixed_dataset_balanced_16k(123456).pkl")


def load_dataframe(path: Path) -> pd.DataFrame:
    with path.open("rb") as f:
        df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path} did not contain a pandas DataFrame")
    expected_cols = {"H", "power", "label"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"{path} missing required columns {expected_cols}")
    return df


def pad_matrix(mat: np.ndarray, cols: int) -> np.ndarray:
    padded = np.zeros((M, K_MAX))
    padded[:, :cols] = mat
    return padded


def main() -> None:
    records = []
    rng = np.random.default_rng(42)

    preview_printed = False

    for filename, k_users, target_count in SOURCES:
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")

        df = load_dataframe(path)
        if len(df) < target_count:
            raise ValueError(f"{path} only has {len(df)} rows (<{target_count})")

        sampled_idx = rng.choice(len(df), size=target_count, replace=False)
        sampled_df = df.iloc[sampled_idx]
        print(
            f"{filename}: selected {target_count} rows out of {len(df)} "
            f"for K={k_users}"
        )

        mask = [1] * k_users + [0] * (K_MAX - k_users)

        for _, row in sampled_df.iterrows():
            H = np.array(row["H"])
            W = np.array(row["label"])
            H_pad = pad_matrix(H, k_users)
            W_pad = pad_matrix(W, k_users)
            records.append(
                {
                    "H": H_pad.tolist(),
                    "label": W_pad.tolist(),
                    "mask": mask.copy(),
                    "power": row["power"],
                    "K": k_users,
                }
            )
            if not preview_printed:
                preview_printed = True
                print("Preview of first padded sample:")
                print("K:", k_users, "mask:", mask)
                print("H_pad:\n", np.array_str(H_pad, precision=6, suppress_small=True))
                print("W_pad:\n", np.array_str(W_pad, precision=6, suppress_small=True))

    df_mixed = pd.DataFrame(records)
    print(f"Total mixed samples: {len(df_mixed)}")
    df_mixed.to_pickle(OUTPUT_PATH)
    print(f"Saved balanced mixed dataset (DataFrame) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
