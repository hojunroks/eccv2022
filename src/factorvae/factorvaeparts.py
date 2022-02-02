import torch.nn as nn
import torch.nn.init as init


class FactorVAE2(nn.Module):
    """Encoder and Decoder architecture for 3D Shapes, Celeba, Chairs data."""
    def __init__(self, z_dim=128):
        super(FactorVAE2, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            EncoderBlock(3,32),
            EncoderBlock(32,32),
            EncoderBlock(32,32),
            EncoderBlock(32,64),
            EncoderBlock(64,64),
            EncoderBlock(64,128),
            EncoderBlock(128,128),
            nn.Conv2d(128, 2*z_dim, 1)
        )
        self.decode = nn.Sequential(
            nn.Conv2d(z_dim, 128, 1),
            nn.ReLU(True),
            DecoderBlock(128, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 64),
            DecoderBlock(64, 32),
            DecoderBlock(32, 32),
            DecoderBlock(32, 32),
            DecoderBlock(32, 3),
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



class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
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

    def forward(self, z):
        return self.net(z).squeeze()


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, 4, 2, 1)
        self.b1 = nn.BatchNorm2d(out_channels)
        self.c2 = nn.Conv2d(out_channels, out_channels, 3, padding="same")
        self.b2 = nn.BatchNorm2d(out_channels)
        self.c3 = nn.Conv2d(out_channels, out_channels, 3, padding="same")
        self.b3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = nn.ReLU(True)(self.b1(self.c1(x)))
        y = x
        x = nn.ReLU(True)(self.b2(self.c2(x)))
        x = nn.ReLU(True)(self.b3(self.c3(x)))
        x = y+x
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1 = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        self.b1 = nn.BatchNorm2d(out_channels)
        self.c2 = nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1)
        self.b2 = nn.BatchNorm2d(out_channels)
        self.c3 = nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1)
        self.b3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = nn.ReLU(True)(self.b1(self.c1(x)))
        y = x
        x = nn.ReLU(True)(self.b2(self.c2(x)))
        x = nn.ReLU(True)(self.b3(self.c3(x)))
        x = y+x
        return x