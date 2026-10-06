import torch
import torch.nn as nn
from data.dataset import Dataset
from torch.utils.data import DataLoader

train = Dataset(csv_path='data/diabetes.csv', split = "train")
val = Dataset(csv_path='data/diabetes.csv', split = "val")
test = Dataset(csv_path='data/diabetes.csv', split = "test")

train_loader = DataLoader(dataset=train, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=train, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset=train, batch_size=32, shuffle=False)

class Model(nn.Module):
    def __init__(self, input, hidden, output):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input, hidden)
        self.relu = nn.ReLU
        self.layer2 = nn.Linear(hidden, output)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
