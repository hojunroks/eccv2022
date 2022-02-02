import random
import pytorch_lightning as pl
import pandas as pd
import torch
import os
import numpy as np
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

class CelebAEncodedDataset(Dataset):
    def __init__(self, root, train=0):
        partition =  pd.read_csv(os.path.join(root, 'list_eval_partition.csv'))
        indices = partition.index[partition['partition']==train].tolist()
        self.attributes = pd.read_csv(os.path.join(root, 'list_attr_celeba.csv')).iloc[indices]
        self.encoded_dir = os.path.join('data_encoded')
        
    def __len__(self):
        return len(self.attributes)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        encoding_name = os.path.join(self.encoded_dir, self.attributes.iloc[idx, 0]).replace('jpg', 'pt')

        encoded_tensor = torch.load(encoding_name).cpu()[0]
        attributes = self.attributes.iloc[idx, 1:].to_numpy(dtype=np.float32)
        attributes = torch.from_numpy(attributes)
        return encoded_tensor,attributes


class CelebAEncodedData(pl.LightningDataModule):
    def __init__(self, args):   
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.image_size = 128

    def train_dataloader(self):
        dataset = CelebAEncodedDataset(root=self.data_dir, train=0)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True
        )
        return dataloader

    def val_dataloader(self):
        dataset = CelebAEncodedDataset(root=self.data_dir, train=1)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True
        )
        return dataloader

    def test_dataloader(self):
        dataset = CelebAEncodedDataset(root=self.data_dir, train=2)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True
        )
        return dataloader


class CelebADataset(Dataset):
    def __init__(self, root, train=0, transform=None):
        partition =  pd.read_csv(os.path.join(root, 'list_eval_partition.csv'))
        indices = partition.index[partition['partition']==train].tolist()
        self.attributes = pd.read_csv(os.path.join(root, 'list_attr_celeba.csv')).iloc[indices]
        self.images_dir = os.path.join(root, 'img_align_celeba')
        self.transform = transform
    
    def __len__(self):
        return len(self.attributes)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.images_dir, self.attributes.iloc[idx, 0])
        image = io.imread(img_name)
        attributes = self.attributes.iloc[idx, 1:].to_numpy(dtype=np.float32)

        if self.transform:
            image = self.transform(image)
        
        attributes = torch.from_numpy(attributes)
        return image,attributes


class CelebAData(pl.LightningDataModule):
    def __init__(self, args):   
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.image_size = 128
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((self.image_size, self.image_size)),
                # T.RandomCrop(self.image_size, padding=8),
                # T.RandomHorizontalFlip(),
                T.ToTensor(),
                # T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CelebADataset(root=self.data_dir, train=0, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                # T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CelebADataset( root=self.data_dir, train=1, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True
        )
        return dataloader

    def test_dataloader(self):
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                # T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CelebADataset( root=self.data_dir, train=2, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True
        )
        return dataloader



class CelebAResizedDataset(Dataset):
    def __init__(self, root, transform):
        self.attributes = pd.read_csv(os.path.join(root, 'list_attr_celeba.csv'))
        self.images_dir = os.path.join(root, 'img_align_celeba_128')
        self.indices = range(len(self))
        self.transform = transform


    def __len__(self):
        return len(self.attributes)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.images_dir, self.attributes.iloc[idx, 0])
        image = io.imread(img_name)
        idx2 = random.choice(self.indices)
        img_name2 = os.path.join(self.images_dir, self.attributes.iloc[idx2, 0])
        image2 = io.imread(img_name2)
        if self.transform:
            image = self.transform(image)
            image2 = self.transform(image2)
        return image,image2


class CelebAResizedData(pl.LightningDataModule):
    def __init__(self, args):   
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )
        dataset = CelebAResizedDataset(root=self.data_dir, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True
        )
        return dataloader

