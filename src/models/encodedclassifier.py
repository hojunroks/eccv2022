import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from src.scheduler import WarmupCosineLR
from torchmetrics import Accuracy, AUROC

class EncodedClassifierNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
        )
        

    def forward(self, x):
        x = self.s(x)
        return x


class EncodedClassifier(pl.LightningModule):
    #################################################################################
    # A simple classifier. 
    #################################################################################
    def __init__(self, hparams, model, target_attr, preEncoder=None, translator=None, *args, **kwargs):
        super().__init__()
        self.hparams.update(vars(hparams))
        self.model = EncodedClassifierNetwork()
        self.target_attr = target_attr
        self.translator=translator
        self.p = 0.5
        self.accuracy = Accuracy()
        self.auroc = AUROC(num_classes=2)

    def forward(self, x):
        x = self.model(x)
        return x

    def shared_step(self, batch, batch_index):
        x, y = batch
        y_hat = self.forward(x)
        y_true = ((y[:,self.target_attr]+1)/2).long()
        loss = F.cross_entropy(y_hat, y_true)
        accuracy = self.accuracy(y_hat, y_true)
        auroc = self.auroc(y_hat, y_true)
        return loss, accuracy, auroc

    def training_step(self, batch, batch_index):
        loss, accuracy, auroc = self.shared_step(batch, batch_index)
        self.log('loss/train', loss)
        self.log('acc/train', accuracy)
        self.log('auroc/train', auroc)
        return loss

    def validation_step(self, batch, batch_index):
        loss, accuracy, auroc = self.shared_step(batch, batch_index)
        self.log('loss/val', loss)
        self.log('acc/val', accuracy)
        self.log('auroc/val', auroc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        total_steps = self.hparams.max_epochs * len(self.trainer._data_connector._train_dataloader_source.dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.05, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "lr",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, required=False)
        parser.add_argument("--weight_decay", type=float, required=False)
        return parser
        
