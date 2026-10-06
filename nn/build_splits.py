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
df["split"] = ["train"] * int(0.7 * n) + ["val"] * int(0.15 * n) + ["test"] * int(0.15 * n)






