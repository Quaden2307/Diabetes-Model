import pandas as pd
import numpy as np
import torch as torch
import csv
from pathlib import Path
from data.dataset import Dataset

DATA_DIR = Path('data/diabetes.csv')

df = pd.read_csv(DATA_DIR)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

n = len(df)
n_train = int(0.7 * n)
n_val = int(0.15 * n)
n_test = n - n_train - n_val

df["split"] = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

train = df["split" == "train"]
val = df[df["split"] == "val"]
test = df[df["split"] == "test"]



print(train)



