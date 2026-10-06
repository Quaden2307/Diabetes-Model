import torch
import pandas as pd
from pathlib import Path

features = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age']

class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, split):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data['split'] == split]
        self.X = torch.tensor(self.data[features].values, dtype=torch.float32)
        self.y = torch.tensor(self.data['Outcome'].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.X[index], self.y[index]



        
        

