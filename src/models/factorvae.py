import pytorch_lightning as pl
import torch.nn.init as init
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
from src.factorvae.ops import recon_loss, kl_divergence, permute_dims


class FactorVAE(pl.LightningModule):
    ##################################################
    # FactorVAE for disentagled representations
    ##################################################
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(vars(hparams))
        self.vae = FactorVAE2()
        # self.D = Discriminator()

    def forward(self, x):
        
        
        pass
    
    def training_step(self, batch, batch_index, optimizer_index):
        x_true1, x_true2 = batch
        x_recon, mu, logvar, z = self.vae(x_true1)
        D_z = self.D(z)
        
        if optimizer_index==0:
            vae_recon_loss = recon_loss(x_true1, x_recon)
            vae_kld = kl_divergence(mu, logvar)
            vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()
            vae_loss = vae_recon_loss + vae_kld + self.hparams.gamma*vae_tc_loss
            return vae_loss
        elif optimizer_index==1:
            z_prime = self.vae(x_true2, no_dec=True)
            z_pperm = permute_dims(z_prime).detach()
            D_z_pperm = self.D(z_pperm)
            D_tc_loss = 0.5*(F.cross_entropy(D_z, self.zeros) + F.cross_entropy(D_z_pperm, self.ones))
            D_tc_loss.backward()
            return D_tc_loss
            
        

    def configure_optimizers(self):
        return super().configure_optimizers()






class FactorVAE2(nn.Module):
    """Encoder and Decoder architecture for 3D Shapes, Celeba, Chairs data."""
    def __init__(self, z_dim=10):
        super(FactorVAE2, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 2*z_dim, 1)
        )
        self.decode = nn.Sequential(
            nn.Conv2d(z_dim, 256, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode(z)
            return x_recon, mu, logvar, z.squeeze()



def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)




class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, 3)
        self.b1 = nn.InstanceNorm2d(out_channels)
        self.c2 = nn.Conv2d(out_channels, out_channels, 3, padding="same")
        self.b2 = nn.InstanceNorm2d(out_channels)
        self.c3 = nn.Conv2d(out_channels, out_channels, 3, padding="same")
        self.b3 = nn.InstanceNorm2d(out_channels)
    
    def forward(self, x):
        x = nn.ReLU()(self.b1(self.c1(x)))
        y = x
        x = nn.ReLU()(self.b2(self.c2(x)))
        x = nn.ReLU()(self.b3(self.c3(x)))
        x = y+x
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1 = nn.ConvTranspose2d(in_channels, out_channels, 3)
        self.b1 = nn.InstanceNorm2d(out_channels)
        self.c2 = nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1)
        self.b2 = nn.InstanceNorm2d(out_channels)
        self.c3 = nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1)
        self.b3 = nn.InstanceNorm2d(out_channels)
    
    def forward(self, x):
        x = nn.ReLU()(self.b1(self.c1(x)))
        y = x
        x = nn.ReLU()(self.b2(self.c2(x)))
        x = nn.ReLU()(self.b3(self.c3(x)))
        x = y+x
        return x