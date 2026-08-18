"""
fetching project dataset and writing it as a plain numeric CSV that
both Python and OCtave can read from.

(since Octave cannot call on sckit-learn)

Run cmd: python export_dataset.py
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing

SEED = 1
TRAIN_FRAC = 0.75
OUT = Path(__file__).with_name("california.csv")

# as_frame=True hands us a DataFrame with 8 features and a target column
# `MedHouseVal` (median house values in $100,000s).
df = fetch_california_housing(as_frame=True).frame

# Shuffle once with a fixed seed
# So positional split is reproducible in both octave and python
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

df.to_csv(OUT, index=False)

n_train = int(TRAIN_FRAC * len(df))
print(f"wrote   : {OUT.name}")
print(f"rows    : {len(df)}")
print(f"columns : {list(df.columns)}")
print(f"nulls   : {int(df.isna().sum().sum())}")
print(f"target  : MedHouseVal (column {len(df.columns)}), $100,000s")
print()
print("Shared split (for both tools):")
print(f" train = first {n_train} data rows")
print(f" test  = last {len(df) - n_train} data rows")
print()
print("Octaver reads it with: data = csvread('california.csv', 1, 0);")
