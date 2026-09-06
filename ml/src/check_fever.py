import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from fever_loader import load_fever_raw
from preprocessing import clean_text

df, _ = load_fever_raw("ml/data/raw/fever/train.jsonl")
df["claim_clean"] = df["claim"].apply(clean_text)

label_nuni = df.groupby("claim_clean")["label"].nunique()
conflicting_claims = set(label_nuni[label_nuni > 1].index)

df_conflicts = df[df["claim_clean"].isin(conflicting_claims)]
df_clean = df[~df["claim_clean"].isin(conflicting_claims)]

print(f"Conflicting unique claims: {len(conflicting_claims):,}")
print(f"Rows occupied by conflicting claims: {len(df_conflicts):,}")
print(f"Clean rows remaining before deduplication: {len(df_clean):,}")
print(f"Clean unique claims: {df_clean['claim_clean'].nunique():,}")
