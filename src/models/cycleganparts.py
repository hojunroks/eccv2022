import torch
import torch.nn as nn
from torch.nn.modules.pooling import MaxPool2d


class CycleGanCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Sequential(
            nn.Conv2d(512, 256, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        x = self.s(x)
        return x

class CycleGanGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Sequential(
            nn.Conv2d(512, 256, 2),
            nn.InstanceNorm2d(256),
            nn.ReLU()
        )
        self.e2 = nn.Sequential(
            nn.Conv2d(256, 256, 2),
            nn.InstanceNorm2d(256),
            nn.ReLU()
        )
        self.e3 = nn.Sequential(   
            nn.Conv2d(256, 1024, 2),
            nn.InstanceNorm2d(1024),
        )
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 2),
            nn.InstanceNorm2d(256),
            nn.ReLU()
        )
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2),
            nn.InstanceNorm2d(256),
            nn.ReLU()
        )
        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 2),
            nn.InstanceNorm2d(512),
        )
        
        self.c1 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU()
        )
        
        self.c2 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU()
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
        )
        
    def forward(self, x):
        o = x
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        x = self.d1(e3)
        x = self.d2(self.c1(torch.cat((x,e2), dim=1)))
        x = self.d3(self.c2(torch.cat((x,e1), dim=1)))
        x = self.c3(torch.cat((x,o), dim=1))
        return x



class CycleGanCriticFC(nn.Module):
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

class CycleGanGeneratorFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = FCBlock(1024, 1024)
        self.f5 = FCBlock(1024, 1024)
        self.f6 = nn.Linear(1024,1024)
        self.b = nn.BatchNorm1d(1024)

    def forward(self, x):
        x1 = self.f1(x)
        x5 = self.f5(x1)
        x6 = self.b(self.f6(x5))
        return x5+x6


class FCBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.f1 = nn.Linear(in_channels, out_channels)
        self.b1 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        x = nn.ReLU()(self.b1(self.f1(x))+x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, 2)
        self.b1 = nn.InstanceNorm2d(out_channels)
        self.c2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.b2 = nn.InstanceNorm2d(out_channels)
        self.c3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.b3 = nn.InstanceNorm2d(out_channels)
    
    def forward(self, x):
        x = nn.LeakyReLU(0.02)(self.b1(self.c1(x)))
        y = x
        x = nn.LeakyReLU(0.02)(self.b2(self.c2(x)))
        x = self.b3(self.c3(x))
        x = y+x
        return nn.LeakyReLU(0.02)(x)

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
