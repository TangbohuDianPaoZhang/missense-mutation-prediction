import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = pd.read_csv('dataset/traindata.csv')
column_to_keep = ['MetaRNN_rankscore', 'BayesDel_noAF_rankscore', 'REVEL_rankscore',
                  'VEST4_rankscore', 'MutPred_rankscore']
features = data[column_to_keep]
target = data['True Label']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.int64).to(device)
y_test = torch.tensor(y_test.values, dtype=torch.int64).to(device)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class Model(nn.Module):
    def __init__(self, input_dim, model_dim, num_classes):
        super(Model, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=2)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc = nn.Linear(model_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x.unsqueeze(1))
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
    
input_dim = X_train.shape[1]
model_dim = 64
num_classes = len(data['True Label'].unique())
model = Model(input_dim, model_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

num_epochs = 10
train(model, train_loader, criterion, optimizer, num_epochs, device)

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')

evaluate(model, test_loader, device)
