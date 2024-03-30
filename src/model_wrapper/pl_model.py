import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics import Accuracy


class MyLightningModule(pl.LightningModule):
    def __init__(self, model, task='classification', learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_function = nn.CrossEntropyLoss() if task == 'classification' else nn.MSELoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=model.output_size) if task == 'classification' else nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        ref = batch[0][0]
        alt = batch[0][1]
        annotation = batch[0][2]
        y = batch[1]
        logits =  self.forward(alt) - self.forward(ref)
        loss = self.loss_function(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ref = batch[0][0]
        alt = batch[0][1]
        annotation = batch[0][2]
        y = batch[1]
        logits = self.forward(alt) - self.forward(ref)
        loss = self.loss_function(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)
