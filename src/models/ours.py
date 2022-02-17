from collections import OrderedDict
import itertools
import pytorch_lightning as pl
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from argparse import ArgumentParser
import torch.nn.functional as F
from torchvision.utils import make_grid
from src.models.cycleganparts import CycleGanCritic, CycleGanGenerator, CycleGanCriticFC, CycleGanGeneratorFC
from src.scheduler import WarmupCosineLR
from torchmetrics import Accuracy, AUROC

ATTRIBUTE_KEYS = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']


class OurGan(pl.LightningModule):
    def __init__(self, decoder, hparams, classifier, classifiers, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(vars(hparams))
        

        # self.A2B = CycleGanGeneratorFC
        # self.B2A = CycleGanGeneratorFC
        self.A2B = CycleGanGenerator()
        self.B2A = CycleGanGenerator()

        # self.d_valid = CycleGanCriticFC()
        self.d_valid = CycleGanCritic()
        self.d_attribute = classifier.eval()

        self.classifiers = classifiers

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        self.decoder = decoder

        self.accuracy = Accuracy()
        self.auroc = AUROC(num_classes=2)


    

    def gan_loss(self, y_hat, y):
        return nn.MSELoss()(y_hat, y)

    def cycle_loss(self, y_hat, y):
        return nn.L1Loss()(y_hat, y)

    def identity_loss(self, y_hat, y):
        return nn.L1Loss()(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.d_attribute.eval()
        A_imgs, B_imgs, attribute_as, attribute_bs = batch
        A_imgs = nn.Unflatten(1, (512, 4, 4))(A_imgs)
        B_imgs = nn.Unflatten(1, (512, 4, 4))(B_imgs)

        ################################################
        ##############   Generator Loss  ###############
        ################################################
        fake_A = self.B2A(B_imgs)
        fake_B = self.A2B(A_imgs)
        if optimizer_idx == 0:
            # Identity Loss
            same_B = self.A2B(B_imgs)
            loss_identity_B = self.identity_loss(same_B, B_imgs)*self.hparams.lambda_B*self.hparams.lambda_idt
            same_A = self.B2A(A_imgs)
            loss_identity_A = self.identity_loss(same_A, A_imgs)*self.hparams.lambda_A*self.hparams.lambda_idt

            # GAN Loss
            fake_imgs = torch.cat((fake_A, fake_B), dim=0)
            critic_fake = self.d_valid(fake_imgs)
            real_labels = torch.ones(critic_fake.shape, device=self.device)
            loss_gan = self.gan_loss(critic_fake, real_labels) * self.hparams.lambda_gan * (1-self.hparams.pretrain)

            # Cycle Loss
            recon_A = self.B2A(fake_B)
            loss_ABA_recon = self.cycle_loss(recon_A, A_imgs)*self.hparams.lambda_A
            recon_B = self.A2B(fake_A)
            loss_BAB_recon = self.cycle_loss(recon_B, B_imgs)*self.hparams.lambda_B

            # # Classification Loss
            
            # fakeA_labels = self.d_attribute(torch.flatten(fake_A, start_dim=1))
            # A_labels = torch.ones((fake_A.shape[0]), device=self.device).long()
            # loss_a_ce = F.cross_entropy(fakeA_labels, A_labels)*self.hparams.lambda_ce * (1-self.hparams.pretrain)
            # fakeB_labels = self.d_attribute(torch.flatten(fake_B, start_dim=1))
            # B_labels = torch.zeros((fake_B.shape[0]), device=self.device).long()
            # loss_b_ce = F.cross_entropy(fakeB_labels, B_labels)*self.hparams.lambda_ce * (1-self.hparams.pretrain)

            # Classification Loss
            fakeA_labels = self.d_attribute(torch.flatten(fake_A, start_dim=1))
            A_labels = torch.ones((fake_A.shape[0]), device=self.device).long()
            A_before = self.d_attribute(torch.flatten(A_imgs, start_dim=1))

            fake_a_ce = F.cross_entropy(fakeA_labels, A_labels)
            real_a_ce = F.cross_entropy(A_before, A_labels)
            loss_a_ce = nn.L1Loss()(fake_a_ce.mean(), real_a_ce.mean()) * self.hparams.lambda_ce * (1-self.hparams.pretrain)

            fakeB_labels = self.d_attribute(torch.flatten(fake_B, start_dim=1))
            B_labels = torch.zeros((fake_B.shape[0]), device=self.device).long()
            B_before = self.d_attribute(torch.flatten(B_imgs, start_dim=1))

            fake_b_ce = F.cross_entropy(fakeB_labels, B_labels)
            real_b_ce = F.cross_entropy(B_before, B_labels)
            loss_b_ce = nn.L1Loss()(fake_b_ce.mean(), real_b_ce.mean()) * self.hparams.lambda_ce * (1-self.hparams.pretrain)


            if self.hparams.pretrain:
                generator_loss = loss_identity_B + loss_identity_A + loss_ABA_recon + loss_BAB_recon
            else:
                generator_loss = loss_identity_B + loss_identity_A + loss_gan + loss_ABA_recon + loss_BAB_recon + loss_a_ce + loss_b_ce
            tqdm_dict = {
                "g_loss": generator_loss,
                "id_b_loss": loss_identity_B,
                "id_a_loss": loss_identity_A,
                "gan_loss": loss_gan,
                "recon_a_loss": loss_ABA_recon,
                "recon_b_loss": loss_BAB_recon,
                "ce_a_loss": loss_a_ce,
                "ce_b_loss": loss_b_ce
            }
            for key in tqdm_dict.keys():
                self.log(key, tqdm_dict[key])
            output = OrderedDict({"loss": generator_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output


        ################################################
        ############   Discriminator Loss  #############
        ################################################
        if optimizer_idx == 1:
            ####### d_valid loss ######
            fake_A = self.fake_A_buffer.push_and_pop(fake_A)
            fake_B = self.fake_B_buffer.push_and_pop(fake_B)
            fake_A.detach()
            fake_B.detach()
            fake_imgs = torch.cat((fake_A, fake_B), dim=0)
            real_imgs = torch.cat((A_imgs, B_imgs), dim=0)

            pred_real = self.d_valid(real_imgs)
            real_labels = torch.ones(pred_real.shape,device=self.device)
            loss_d_valid_real = self.gan_loss(pred_real, real_labels)
            pred_fake = self.d_valid(fake_imgs)
            fake_labels = torch.zeros(pred_fake.shape,device=self.device)
            loss_d_valid_fake = self.gan_loss(pred_fake, fake_labels)
            loss_d_valid = loss_d_valid_real + loss_d_valid_fake
            tqdm_dict = {
                "d_valid_loss": loss_d_valid,
                "d_valid_loss_fake": loss_d_valid_fake,
                "d_valid_loss_real": loss_d_valid_real
            }
            for key in tqdm_dict.keys():
                self.log(key, tqdm_dict[key])
            output = OrderedDict({"loss": loss_d_valid, "progress_bar": tqdm_dict, "log": tqdm_dict})            
            return output

    def validation_step(self, batch, batch_idx):
        As, Bs, attribute_as, attribute_bs = batch
        As = nn.Unflatten(1, (512, 4, 4))(As)
        Bs = nn.Unflatten(1, (512, 4, 4))(Bs)
        trues = torch.cat((As,Bs))
        gt = torch.cat((attribute_as, attribute_bs))
        fake_As = self.B2A(Bs)
        fake_Bs = self.A2B(As)
        fakes = torch.cat((fake_Bs, fake_As))
        for cls in self.classifiers.keys():
            self.classifiers[cls].to(self.device)
            gt_labels = ((gt[:, cls]+1)/2).long()
            true_preds = self.classifiers[cls](torch.flatten(trues, start_dim=1))
            fake_preds = self.classifiers[cls](torch.flatten(fakes, start_dim=1))
            accuracy_true = self.accuracy(true_preds, gt_labels)
            auroc_true = self.accuracy(true_preds, gt_labels)
            accuracy_fake = self.accuracy(fake_preds, gt_labels)
            auroc_fake = self.accuracy(fake_preds, gt_labels)
            self.log(ATTRIBUTE_KEYS[cls]+'/accuracy/true', accuracy_true)
            self.log(ATTRIBUTE_KEYS[cls]+'/accuracy/fake', accuracy_fake)
            self.log(ATTRIBUTE_KEYS[cls]+'/auroc/true', auroc_true)
            self.log(ATTRIBUTE_KEYS[cls]+'/auroc/fake', auroc_fake)
                
            
        real_A_imgs = self.decoder(As)
        fake_B_imgs = self.decoder(self.A2B(As))
        reconstructed_A_imgs = self.decoder(self.B2A(self.A2B(As)))
        real_B_imgs = self.decoder(Bs)
        fake_A_imgs = self.decoder(self.B2A(Bs))
        reconstructed_B_imgs = self.decoder(self.A2B(self.B2A(Bs)))
        batch_dictionary={
            "real_As": real_A_imgs,
            "fake_Bs": fake_B_imgs,
            "reconstructed_As": reconstructed_A_imgs,
            "real_Bs": real_B_imgs,
            "fake_As": fake_A_imgs,
            "reconstructed_Bs": reconstructed_B_imgs
        }
        return batch_dictionary

    def validation_epoch_end(self, val_step_outputs):
        idxes = random.sample(range(val_step_outputs[0]['real_As'].shape[0]), min(4,val_step_outputs[0]['real_As'].shape[0]))
        real_As = make_grid(torch.cat([output["real_As"] for output in val_step_outputs])[idxes], nrow=1)
        fake_Bs = make_grid(torch.cat([output["fake_Bs"] for output in val_step_outputs])[idxes], nrow=1)
        reconstructed_As = make_grid(torch.cat([output["reconstructed_As"] for output in val_step_outputs])[idxes], nrow=1)
        real_Bs = make_grid(torch.cat([output["real_Bs"] for output in val_step_outputs])[idxes], nrow=1)
        fake_As = make_grid(torch.cat([output["fake_As"] for output in val_step_outputs])[idxes], nrow=1)
        reconstructed_Bs = make_grid(torch.cat([output["reconstructed_Bs"] for output in val_step_outputs])[idxes], nrow=1)
        fig = plt.figure(figsize=(18, 12))
        fig.add_subplot(1,6,1)
        plt.axis('off')
        plt.imshow(real_As.permute(1,2,0).data.cpu().numpy())
        fig.add_subplot(1,6,2)
        plt.axis('off')
        plt.imshow(fake_Bs.permute(1,2,0).data.cpu().numpy())
        fig.add_subplot(1,6,3)
        plt.axis('off')
        plt.imshow(reconstructed_As.permute(1,2,0).data.cpu().numpy())
        fig.add_subplot(1,6,4)
        plt.axis('off')
        plt.imshow(real_Bs.permute(1,2,0).data.cpu().numpy())
        fig.add_subplot(1,6,5)
        plt.axis('off')
        plt.imshow(fake_As.permute(1,2,0).data.cpu().numpy())
        fig.add_subplot(1,6,6)
        plt.axis('off')
        plt.imshow(reconstructed_Bs.permute(1,2,0).data.cpu().numpy())
        plt.tight_layout()
        
        self.logger.experiment.add_figure('figure', fig, global_step=self.current_epoch)
        return

    def configure_optimizers(self):
        total_steps = self.hparams.max_epochs * len(self.trainer._data_connector._train_dataloader_source.dataloader())
        lr_g = self.hparams.lr_g
        lr_d = self.hparams.lr_d
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(itertools.chain(self.A2B.parameters(), self.B2A.parameters()), lr=lr_g, betas=(b1, b2))
        opt_d_valid = torch.optim.Adam(itertools.chain(self.d_valid.parameters()), lr=lr_d, betas=(b1, b2))
        scheduler_g = {
            "scheduler": WarmupCosineLR(
                opt_g, warmup_epochs=total_steps * 0.05, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "lr_g",
        }
        scheduler_d_valid = {
            "scheduler": WarmupCosineLR(
                opt_d_valid, warmup_epochs=total_steps * 0.05, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "lr_d",
        }
        return [opt_g, opt_d_valid]  , [scheduler_g, scheduler_d_valid]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr_g", type=float, required=False)
        parser.add_argument("--lr_d", type=float, required=False)
        parser.add_argument("--b1", type=float, required=False)
        parser.add_argument("--b2", type=float, required=False)
        parser.add_argument("--lambda_A", type=float, required=False)
        parser.add_argument("--lambda_B", type=float, required=False)
        parser.add_argument("--lambda_idt", type=float, required=False)
        parser.add_argument("--lambda_ce", type=float, required=False)
        parser.add_argument("--lambda_gan", type=float, required=False)
        parser.add_argument("--pretrain", type=int, required=False)
        parser.add_argument("--classifiers", type=list, required=False)
        return parser



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