import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torch.nn.modules.container import Sequential
from torchvision import models
from src.scheduler import WarmupCosineLR
from torchmetrics import Accuracy

class AutoEncoder(pl.LightningModule):
    #################################################################################
    # A simple autoencoder. 
    #################################################################################
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(vars(hparams))
        self.input_channels=3
        self.encoder = nn.Sequential(
            EncoderBlock(3, 8),
            EncoderBlock(8, 64),
            EncoderBlock(64, 128),
            EncoderBlock(128, 256)
        )
        self.decoder = nn.Sequential(
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 8),
            DecoderBlock(8, 3)
        )
        self.loss = nn.MSELoss()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_index):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, x)
        self.log('loss/train', loss)
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, x)
        batch_dictionary={
            "orig_img": x,
            "prediction": y_hat,
            "loss": loss
        }
        self.log('loss/validation', loss, on_epoch=True)
        return batch_dictionary
        

    def validation_epoch_end(self, val_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in val_step_outputs]).mean()
        self.logger.experiment.add_scalar("Loss/validation", avg_loss, self.current_epoch)
        orig_images = torch.stack([output["orig_img"][0] for output in val_step_outputs])
        made_images = torch.stack([output["prediction"][0] for output in val_step_outputs])
        idx = random.sample(range(len(orig_images)), min(len(orig_images), 8))
        fig = plt.figure(figsize=(12, 12))
        if len(idx)==8:
            for i in range(4):
                ax = fig.add_subplot(2,4,2*i+1)
                plt.imshow((orig_images[idx[i]].permute(1, 2, 0).cpu()*255).type(torch.int))
                ax = fig.add_subplot(2,4,2*i+2)
                plt.imshow((made_images[idx[i]].permute(1, 2, 0).cpu()*255).type(torch.int))
        plt.tight_layout()
        self.logger.experiment.add_figure('figure', fig, global_step=self.current_epoch)
        return

    def test_step(self, batch, batch_index):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
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
        




class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, 3, padding="same")
        self.b1 = nn.BatchNorm2d(out_channels)
        self.c2 = nn.Conv2d(out_channels, out_channels, 3, padding="same")
        self.b2 = nn.BatchNorm2d(out_channels)
        self.c3 = nn.Conv2d(out_channels, out_channels, 3, padding="same")
        self.b3 = nn.BatchNorm2d(out_channels)
        self.m = nn.MaxPool2d(2)
        
    
    def forward(self, x):
        x = nn.ReLU()(self.b1(self.c1(x)))
        y = x
        x = nn.ReLU()(self.b2(self.c2(x)))
        x = nn.ReLU()(self.b3(self.c3(x)))
        x = y+x
        return self.m(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1 = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
        self.b1 = nn.BatchNorm2d(out_channels)
        self.c2 = nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1)
        self.b2 = nn.BatchNorm2d(out_channels)
        self.c3 = nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1)
        self.b3 = nn.BatchNorm2d(out_channels)
        
        
    
    def forward(self, x):
        x = nn.ReLU()(self.b1(self.c1(x)))
        y = x
        x = nn.ReLU()(self.b2(self.c2(x)))
        x = nn.ReLU()(self.b3(self.c3(x)))
        x = y+x
        return x