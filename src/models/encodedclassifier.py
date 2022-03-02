import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from src.scheduler import WarmupCosineLR
from torchmetrics import Accuracy, AUROC
from sklearn.metrics import confusion_matrix


class EncodedClassifierNetwork(nn.Module):
    def __init__(self, use_dropout):
        super().__init__()
        if use_dropout:
            self.s = nn.Sequential(
                nn.Conv2d(512, 256, 3),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Dropout(),
                nn.Flatten(),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(256, 2),
            )
        else:
            self.s = nn.Sequential(
                nn.Conv2d(512, 256, 3),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 2),
            )
        

    def forward(self, x):
        x = self.s(x)
        return x


class EncodedClassifier(pl.LightningModule):
    #################################################################################
    # A simple classifier. 
    #################################################################################
    def __init__(self, hparams, target_attr, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(vars(hparams))
        self.model = EncodedClassifierNetwork(use_dropout=self.hparams.use_dropout)
        self.target_attr = target_attr
        self.p = 0.5
        self.accuracy = Accuracy()
        self.auroc = AUROC(num_classes=2)

    def forward(self, x):
        x = nn.Unflatten(1, (512, 4, 4))(x)
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
        self.model.train()
        loss, accuracy, auroc = self.shared_step(batch, batch_index)
        self.log('loss/train', loss)
        self.log('acc/train', accuracy)
        self.log('auroc/train', auroc)
        return loss

    def validation_step(self, batch, batch_index):
        self.model.eval()
        x, y = batch
        y_hat = self.forward(x)
        y_true = ((y[:,self.target_attr]+1)/2).long()
        print(y_true)
        loss = F.cross_entropy(y_hat, y_true)
        accuracy = self.accuracy(y_hat, y_true)
        auroc = self.auroc(y_hat, y_true)
        self.log('loss/val', loss)
        self.log('acc/val', accuracy)
        self.log('auroc/val', auroc)
        return confusion_matrix(y_true.cpu(), torch.argmax(y_hat, 1).cpu())
    
    def validation_epoch_end(self, outputs):
        print(outputs)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        # optimizer = torch.optim.SGD(
        #     self.model.parameters(), 
        #     lr=self.hparams.lr, 
        #     momentum=0.9,
        #     weight_decay=self.hparams.weight_decay
        # )
        total_steps = self.hparams.max_epochs * len(self.trainer._data_connector._train_dataloader_source.dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.05, max_epochs=total_steps/self.hparams.gpus
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
        parser.add_argument("--use_dropout", type=int, required=False)
        return parser
        
