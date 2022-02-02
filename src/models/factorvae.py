from argparse import ArgumentParser
from collections import OrderedDict
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from src.factorvae.ops import recon_loss, kl_divergence, permute_dims
from src.factorvae.factorvaeparts import FactorVAE2, Discriminator

class FactorVAE(pl.LightningModule):
    ##################################################
    # FactorVAE for disentagled representations
    ##################################################
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(vars(hparams))
        print(self.hparams.z_dim)
        self.vae = FactorVAE2(z_dim=self.hparams.z_dim)
        self.D = Discriminator(self.hparams.z_dim)
        self.zeros = torch.zeros(self.hparams.batch_size).to(self.device)
        self.ones = torch.ones(self.hparams.batch_size).to(self.device)
        
    def training_step(self, batch, batch_index, optimizer_idx):
        x_true1, x_true2 = batch
        x_recon, mu, logvar, z = self.vae(x_true1)
        D_z = self.D(z.squeeze())
        
        if optimizer_idx==0:
            vae_recon_loss = recon_loss(x_true1, x_recon)
            vae_kld = kl_divergence(mu, logvar)
            vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()
            vae_loss = vae_recon_loss + vae_kld + self.hparams.gamma*vae_tc_loss
            tqdm_dict = {
                "vae/loss": vae_loss,
                "vae/recon_loss": vae_recon_loss,
                "vae/kld": vae_kld,
                "vae/tc_loss": vae_tc_loss
            }
            for key in tqdm_dict.keys():
                self.log(key, tqdm_dict[key])
            output = OrderedDict({"loss": vae_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output
            
        elif optimizer_idx==1:
            z_prime = self.vae(x_true2, no_dec=True)
            z_pperm = permute_dims(z_prime).detach().squeeze()
            D_z_pperm = self.D(z_pperm)
            D_tc_loss = 0.5*(F.cross_entropy(D_z, self.zeros) + F.cross_entropy(D_z_pperm, self.ones))
            self.log("D/tc_loss", D_tc_loss)
            return D_tc_loss
            
    def configure_optimizers(self):
        opt_vae = optim.Adam(self.vae.parameters(), lr=self.hparams.lr_vae, betas = (self.hparams.beta1_vae, self.hparams.beta2_vae))
        d_vae = optim.Adam(self.D.parameters(), lr=self.hparams.lr_D, betas = (self.hparams.beta1_D, self.hparams.beta2_D))
        return [opt_vae, d_vae]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr_vae", type=float, required=False)
        parser.add_argument("--beta1_vae", type=float, required=False)
        parser.add_argument("--beta2_vae", type=float, required=False)
        parser.add_argument("--lr_D", type=float, required=False)
        parser.add_argument("--beta1_D", type=float, required=False)
        parser.add_argument("--beta2_D", type=float, required=False)
        parser.add_argument("--gamma", type=float, required=False)
        parser.add_argument("--z_dim", type=int, required=False)
        return parser