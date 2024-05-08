import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics import Accuracy
from torchmetrics import AUROC


class MyLightningModuleDelta(pl.LightningModule):
    def __init__(self, model, task='classification', learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_function = nn.CrossEntropyLoss() if task == 'classification' else nn.MSELoss()
        self.task = task
        if task == 'classification':
            # warning: AUROC is very slow to compute, so it should be used sparingly for test set evaluation
            self.accuracy = Accuracy(task="multiclass", num_classes=model.output_size)
            self.test_accuracy = AUROC(task="multiclass", num_classes=model.output_size)
        elif task == 'regression':
            self.accuracy = nn.MSELoss()
            self.test_accuracy = nn.MSELoss()
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        seq = batch[0][0]
        annotation = batch[0][1]
        y = batch[1].squeeze(1)
        logits =  self.forward(seq)
        loss = self.loss_function(logits, y)
        if self.task == 'classification':
            preds = torch.argmax(logits, dim=1)
        else:
            preds = logits
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seq = batch[0][0]
        annotation = batch[0][1]
        y = batch[1].squeeze(1)
        logits =  self.forward(seq)
        loss = self.loss_function(logits, y)
        if  self.task == 'classification':
            preds = torch.argmax(logits, dim=1)
        else:
            preds = logits
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        seq = batch[0][0]
        annotation = batch[0][1]
        y = batch[1]
        # logits = self.forward(alt) - self.forward(ref)
        logits =  self.forward(seq)
        loss = self.loss_function(logits, y)
        if self.task == 'classification':
            preds_auc = torch.sigmoid(logits)
            auc = self.test_accuracy(preds_auc, y)
            self.log('test_auc', auc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            preds = torch.argmax(logits, dim=1)
        else:
            preds = logits
        acc = self.test_accuracy(preds, y)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)
