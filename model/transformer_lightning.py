import os
import pandas as pd
from torch import optim, nn, utils, Tensor
import torch
from sklearn.model_selection import train_test_split
import lightning as L


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


class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer
    

# 初始化模型实例
input_dim = 5
model_dim = 128
num_classes = 2  # 根据实际类别数量设置
model = Model(input_dim=input_dim, model_dim=model_dim, num_classes=num_classes)
lit_model = LitModel(model)

# 读取数据
data = pd.read_csv('dataset/traindata.csv')
column_to_keep = ['MetaRNN_rankscore', 'BayesDel_noAF_rankscore', 'REVEL_rankscore',
                  'VEST4_rankscore', 'MutPred_rankscore']
features = data[column_to_keep]
target = data['True Label']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 转换为 tensor
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.int64)
y_test = torch.tensor(y_test.values, dtype=torch.int64)

# 创建 DataLoader
train_dataset = utils.data.TensorDataset(X_train, y_train)
test_dataset = utils.data.TensorDataset(X_test, y_test)
train_loader = utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 设置 Trainer 并开始训练
trainer = L.Trainer(limit_train_batches=100, max_epochs=5, devices=1, accelerator="gpu")
trainer.fit(model=lit_model, train_dataloaders=train_loader)
