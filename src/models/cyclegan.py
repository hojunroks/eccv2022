from collections import OrderedDict
import itertools
import pytorch_lightning as pl
import torch
import torch.nn as nn
import random
from argparse import ArgumentParser
import gc
from src.models.cycleganparts import CycleGanCritic, CycleGanGenerator

class CycleGan(pl.LightningModule):
    def __init__(self, lr=1e-4, b1=0.5, b2=0.999, *args, **kwargs):
        super().__init__()
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.save_hyperparameters()
        self.target_attr = self.hparams.target_attr

        self.A2B = CycleGanGenerator()
        self.B2A = CycleGanGenerator()

        self.d_A = CycleGanCritic()
        self.d_B = CycleGanCritic()

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    def gan_loss(self, y_hat, y):
        return torch.mean(y_hat)-torch.mean(y)

    def cycle_loss(self, y_hat, y):
        return nn.L1Loss()(y_hat, y)

    def identity_loss(self, y_hat, y):
        return nn.L1Loss()(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        gc.collect()
        torch.cuda.empty_cache()
        image_batch, attributes_batch = batch
        target_attrs = (attributes_batch[:, self.target_attr]+1)/2
        A_idxes = torch.nonzero(target_attrs)[:,0]
        B_idxes = torch.nonzero(1-target_attrs)[:,0]
        smaller_size = min(A_idxes.shape[0], B_idxes.shape[0])
        A_idxes = A_idxes[:smaller_size]
        B_idxes = B_idxes[:smaller_size]
        A_imgs = image_batch[A_idxes]
        B_imgs = image_batch[B_idxes]

        ################################################
        ##############   Generator Loss  ###############
        ################################################
        fake_A = self.B2A(B_imgs)
        fake_B = self.A2B(A_imgs)
        if optimizer_idx == 0:
            # Identity Loss
            same_B = self.A2B(B_imgs)
            loss_identity_B = self.identity_loss(same_B, B_imgs)*5.0
            same_A = self.B2A(A_imgs)
            loss_identity_A = self.identity_loss(same_A, A_imgs)*5.0

            # GAN Loss
            
            critic_fake_B = self.d_B(fake_B)
            loss_gan_A2B = -torch.mean(critic_fake_B)
            
            critic_fake_A = self.d_A(fake_A)
            loss_gan_B2A = -torch.mean(critic_fake_A)
            
            # Cycle Loss
            recon_A = self.B2A(fake_B)
            loss_ABA_recon = self.cycle_loss(recon_A, A_imgs)
            recon_B = self.A2B(fake_A)
            loss_BAB_recon = self.cycle_loss(recon_B, B_imgs)

            generator_loss = loss_identity_B + loss_identity_A + loss_gan_A2B + loss_gan_B2A + loss_ABA_recon + loss_BAB_recon
            tqdm_dict = {
                "g_loss": generator_loss,
                "id_b_loss": loss_identity_B,
                "id_a_loss": loss_identity_A,
                "gan_ab_loss": loss_gan_A2B,
                "gan_ba_loss": loss_gan_B2A,
                "recon_a_loss": loss_ABA_recon,
                "recon_b_loss": loss_BAB_recon
            }
            for key in tqdm_dict.keys():
                self.log(key, tqdm_dict[key])
            output = OrderedDict({"loss": generator_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output


        ################################################
        ############   Discriminator Loss  #############
        ################################################
        if optimizer_idx == 1:
            ####### d_A loss ######
            fake_A = self.fake_A_buffer.push_and_pop(fake_A)
            grad_penalty_A = self.calc_gradient_penalty(self.d_A, A_imgs, fake_A)
            loss_d_a = self.gan_loss(self.d_A(fake_A), self.d_A(A_imgs)) + grad_penalty_A
            tqdm_dict = {
                "d_a_loss": loss_d_a
            }
            for key in tqdm_dict.keys():
                self.log(key, tqdm_dict[key])
            output = OrderedDict({"loss": loss_d_a, "progress_bar": tqdm_dict, "log": tqdm_dict})            
            return output

        if optimizer_idx == 2:
            ####### d_B loss ######
            fake_B = self.fake_B_buffer.push_and_pop(fake_B)
            grad_penalty_B = self.calc_gradient_penalty(self.d_B, B_imgs, fake_B)
            loss_d_b = self.gan_loss(self.d_B(fake_B), self.d_B(B_imgs)) + grad_penalty_B
            tqdm_dict = {
                "d_b_loss": loss_d_b
            }
            for key in tqdm_dict.keys():
                self.log(key, tqdm_dict[key])
            output = OrderedDict({"loss": loss_d_b, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def configure_optimizers(self):
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(itertools.chain(self.A2B.parameters(), self.B2A.parameters()), lr=1e-4, betas=(b1, b2))
        opt_d_a = torch.optim.Adam(itertools.chain(self.d_A.parameters()), lr=1e-4, betas=(b1, b2))
        opt_d_b = torch.optim.Adam(itertools.chain(self.d_B.parameters()), lr=1e-4, betas=(b1, b2))

        return [opt_g, opt_d_a, opt_d_b] #, [lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B]


    def calc_gradient_penalty(self, critic, real_data, generated_data):
        # GP strength
        LAMBDA = 10

        b_size = real_data.shape[0]

        # Calculate interpolation
        alpha = torch.rand(b_size, 1, 1, 1, requires_grad=True).type_as(real_data)
        alpha = alpha.expand_as(real_data)

        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data

        # Calculate probability of interpolated examples
        prob_interpolated = critic(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
                outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.shape).type_as(real_data),
                            create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(b_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return LAMBDA * ((gradients_norm - 1) ** 2).mean()


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, required=False)
        parser.add_argument("--b1", type=float, required=False)
        parser.add_argument("--b2", type=float, required=False)
        parser.add_argument("--target_attr", type=int, required=False)
        
        return parser





class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)
