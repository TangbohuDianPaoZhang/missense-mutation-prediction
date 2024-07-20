import pandas as pd
import torch
import lightning as L
from sklearn.metrics import accuracy_score
from torch import optim, nn, utils
from torchmetrics import Accuracy, AUROC


class Model(nn.Module):
    def __init__(self, input_dim, model_dim, num_classes):
        super(Model, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=2048,
            dropout=0.1)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=4)
        self.fc = nn.Linear(model_dim, num_classes)
    
    def forward(self, src):
        x = self.embedding(src)
        x = self.transformer(x.unsqueeze(1))
        x = x.squeeze(1)
        x = self.fc(x)
        return x


class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='binary')
        self.auroc = AUROC(task='binary')
        self.validation_step_outputs = []


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y = nn.functional.one_hot(y, num_classes=2).float()
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):  
        x, y = batch
        y = nn.functional.one_hot(y, num_classes=2).float()  
        y_hat = self.model(x)  
        val_loss = self.criterion(y_hat, y)
        self.validation_step_outputs.append(val_loss)
        self.log('val_loss', val_loss, on_step=True, prog_bar=True)
        self.accuracy.update(y_hat, y)
        self.auroc.update(y_hat, y)

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        val_accuracy = self.accuracy.compute()
        val_auroc = self.auroc.compute()
        self.log('val_accuracy', val_accuracy, prog_bar=True)
        self.log('val_auroc', val_auroc, prog_bar=True)
        self.accuracy.reset()
        self.auroc.reset()
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** (epoch // 3))
        return [optimizer], [scheduler]
    

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Log learning rate
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, on_epoch=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = nn.functional.one_hot(y, num_classes=2).float()
        y_hat = self.model(x)
        return {'preds': torch.argmax(y_hat, dim=1), 'target': y}
    
    def on_test_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs])
        targets = torch.cat([x['target'] for x in outputs])
        acc = accuracy_score(targets.cpu(), preds.cpu())
        self.log('test_acc', acc)
        return acc
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        y_hat = self.model(x)
        return y_hat
    

train_data_raw = pd.read_csv('dataset/train_dataset_filled(n=3).csv')
train_feature = train_data_raw.drop('True Label', axis=1)
train_target = train_data_raw['True Label']
train_tensor = torch.tensor(train_feature.values, dtype=torch.float32)
target_tensor = torch.tensor(train_target.values, dtype=torch.int64)

test_data_raw = pd.read_csv('dataset/test_dataset_filled(n=3).csv')
test_feature = test_data_raw.drop('True Label', axis=1)
test_target = test_data_raw['True Label']
test_tensor = torch.tensor(test_feature.values, dtype=torch.float32)
test_target_tensor = torch.tensor(test_target.values, dtype=torch.int64)

input_dim = train_feature.shape[1]
model_dim = 256
num_classes = 2  # 根据实际类别数量设置
batch_size = 32
model = Model(input_dim=input_dim, model_dim=model_dim, num_classes=num_classes)
lit_model = LitModel(model)

train_dataset = utils.data.TensorDataset(train_tensor, target_tensor)
train_dataloader = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = utils.data.TensorDataset(test_tensor, test_target_tensor)
test_dataloader = utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

trainer = L.Trainer(limit_train_batches=100, max_epochs=15, devices=1, accelerator="gpu", log_every_n_steps=10, enable_checkpointing=True)
trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

# trainer.test(dataloaders=test_dataloader)
