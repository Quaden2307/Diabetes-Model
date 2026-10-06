import pandas as pd
from pathlib import Path

DATA_DIR = Path('data/diabetes.csv')

df = pd.read_csv(DATA_DIR)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

n = len(df)
n_train = int(0.7 * n)
n_val = int(0.15 * n)
n_test = n - n_train - n_val

df["split"] = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

df.to_csv("data/diabetes_splits.csv", index=False)



