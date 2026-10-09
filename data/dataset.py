import torch
import pandas as pd

features = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age']

class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, split):
        self.data = pd.read_csv(csv_path)
        # normalization stats always come from the train rows, never from val/test
        train_rows = self.data[self.data['split'] == 'train']
        mean = torch.tensor(train_rows[features].mean().values, dtype=torch.float32)
        std = torch.tensor(train_rows[features].std().values, dtype=torch.float32)
        self.data = self.data[self.data['split'] == split]
        self.X = torch.tensor(self.data[features].values, dtype=torch.float32)
        self.X = (self.X - mean) / std
        self.y = torch.tensor(self.data['Outcome'].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.X[index], self.y[index]



        
        

