import pytorch_lightning as pl
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torchvision.utils import make_grid
from src.scheduler import WarmupCosineLR

class AutoEncoder(pl.LightningModule):
    #################################################################################
    # A simple autoencoder. 
    #################################################################################
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(vars(hparams))
        self.encoder = nn.Sequential(
            EncoderBlock(3, 8),
            EncoderBlock(8, 32),
            EncoderBlock(32, 64),
            EncoderBlock(64, 128),
            EncoderBlock(128, 256),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 32),
            DecoderBlock(32, 8),
            DecoderBlock(8, 3),
            nn.Sigmoid()
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
        orig_images = torch.cat([output["orig_img"] for output in val_step_outputs])
        made_images = torch.cat([output["prediction"] for output in val_step_outputs])
        idxes = random.sample(range(orig_images.shape[0]), 25)
        orig_images = make_grid(orig_images[idxes], nrow=5)
        made_images = make_grid(made_images[idxes], nrow=5)
        fig = plt.figure(figsize=(12, 12))
        fig.add_subplot(1,2,1)
        plt.axis('off')
        plt.imshow(orig_images.permute(1,2,0).data.cpu().numpy())
        fig.add_subplot(1,2,2)
        plt.axis('off')
        plt.imshow(made_images.permute(1,2,0).data.cpu().numpy())
        plt.tight_layout()
        
        self.logger.experiment.add_figure('figure', fig, global_step=self.current_epoch)
        return

    def test_step(self, batch, batch_index):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        total_steps = self.hparams.max_epochs * len(self.trainer._data_connector._train_dataloader_source.dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.1, max_epochs=total_steps
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

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.b1 = nn.BatchNorm2d(out_channels)
        self.c2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.b2 = nn.BatchNorm2d(out_channels)
        self.c3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.b3 = nn.BatchNorm2d(out_channels)
        self.m = nn.MaxPool2d(2)
        
    
    def forward(self, x):
        x = nn.ReLU()(self.b1(self.c1(x)))
        y = x
        x = nn.ReLU()(self.b2(self.c2(x)))
        x = self.b3(self.c3(x))
        x = nn.ReLU()(y+x)
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
        x = self.b3(self.c3(x))
        x = nn.ReLU()(y+x)
        return x