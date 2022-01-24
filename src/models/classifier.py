import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from src.scheduler import WarmupCosineLR
from torchmetrics import Accuracy

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Classifier(pl.LightningModule):
    #################################################################################
    # A simple classifier. 
    #################################################################################
    def __init__(self, hparams, model, target_attr, preEncoder=None, *args, **kwargs):
        super().__init__()
        self.hparams.update(vars(hparams))
        self.model = model
        self.target_attr = target_attr
        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Softmax()
        )
        self.accuracy = Accuracy()
        self.preEncoder = Identity()
        if preEncoder!=None:
            self.preEncoder = preEncoder
            self.preEncoder.train()

    def forward(self, x):
        x = self.preEncoder(x)
        x = self.model(x)
        return self.fc(x)

    def shared_step(self, batch, batch_index):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, ((y[:,self.target_attr]+1)/2).long())
        accuracy = self.accuracy(y_hat, ((y[:,self.target_attr]+1)/2).long())
        return loss, accuracy

    def training_step(self, batch, batch_index):
        loss, accuracy = self.shared_step(batch, batch_index)
        self.log('loss/train', loss)
        self.log('acc/train', accuracy)
        return loss

    def validation_step(self, batch, batch_index):
        loss, accuracy = self.shared_step(batch, batch_index)
        self.log('loss/val', loss)
        self.log('acc/val', accuracy)

    def test_step(self, batch, batch_index):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.1, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, required=False)
        parser.add_argument("--weight_decay", type=float, required=False)
        return parser
        
