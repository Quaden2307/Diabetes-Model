import torch
import torch.nn as nn
from data.dataset import Dataset
from torch.utils.data import DataLoader

train = Dataset(csv_path='data/diabetes_splits.csv', split = "train")
val = Dataset(csv_path='data/diabetes_splits.csv', split = "val")
test = Dataset(csv_path='data/diabetes_splits.csv', split = "test")

train_loader = DataLoader(dataset=train, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset=test, batch_size=32, shuffle=False)

class Model(nn.Module):
    def __init__(self, input, hidden, output):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input, hidden)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model = Model(input=8, hidden=16, output=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

epochs = 100
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        preds = model(xb)
        loss = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            val_loss += loss.item()
            correct += (preds.argmax(dim=1) == yb).sum().item()

    print(f"epoch {epoch}: val_loss={val_loss/len(val_loader):.3f}, "
          f"val_acc={correct/len(val):.3f}")




    

    



