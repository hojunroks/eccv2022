from cmath import nan
import random
from attr import attributes
import pytorch_lightning as pl
import pandas as pd
import torch
import os
import numpy as np
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


class CelebACycleganDataset(Dataset):
    def __init__(self, root, partition_csv='list_eval_partition.csv', attributes_csv='list_attr_celeba.csv', train=0, encoded_dir='data_encoded', target_attr='', transform=None):
        partition =  pd.read_csv(os.path.join(root, partition_csv))
        indices = partition.index[partition['partition']==train].tolist()
        self.attributes = pd.read_csv(os.path.join(root, attributes_csv)).iloc[indices].reset_index()
        self.a_indices = self.attributes.index[self.attributes[target_attr]==1].tolist()
        self.b_indices = self.attributes.index[self.attributes[target_attr]==-1].tolist()
        self.a_attributes = self.attributes.iloc[self.a_indices]
        self.b_attributes = self.attributes.iloc[self.b_indices]
        self.encoded_dir = encoded_dir
        
    def __len__(self):
        return max(len(self.a_indices), len(self.b_indices))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx_a = idx
        idx_b = idx
        if(len(self.a_indices)<len(self.b_indices)):
            idx_a = idx_a % len(self.a_indices)
        else:
            idx_b = idx_b % len(self.b_indices)
        
        # print(idx_a)
        # print(idx_b)
        encoded_a = torch.from_numpy(np.load(os.path.join(self.encoded_dir, self.a_attributes.iloc[idx_a, 1]).replace('.jpg', '.npy')))
        attributes_a = torch.from_numpy(self.a_attributes.iloc[idx_a, 2:].to_numpy(dtype=np.float32))

        encoded_b = torch.from_numpy(np.load(os.path.join(self.encoded_dir, self.b_attributes.iloc[idx_b, 1]).replace('.jpg', '.npy')))
        attributes_b = torch.from_numpy(self.b_attributes.iloc[idx_b, 2:].to_numpy(dtype=np.float32))
        
        return encoded_a, encoded_b, attributes_a, attributes_b


class CelebACycleganData(pl.LightningDataModule):
    def __init__(self, args):   
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.target_attr = args.target_attr
        self.encoded_dir = os.path.join(args.encoded_dir, args.pretrained_ver)
    
    def get_dataloader(self, train):
        dataset = CelebACycleganDataset(root=self.data_dir, train=train, target_attr=self.target_attr, encoded_dir=self.encoded_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=(train==0),
            drop_last=True,
            pin_memory=True,
            persistent_workers=True
        )
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader(0)

    def val_dataloader(self):
        return self.get_dataloader(1)

    def test_dataloader(self):
        return self.get_dataloader(2)


class CelebAEncodedDataset(Dataset):
    def __init__(self, root, partition_csv='list_eval_partition.csv', attributes_csv='list_attr_celeba.csv', train=0, encoded_dir='data_encoded', transform=None):
        partition =  pd.read_csv(os.path.join(root, partition_csv))
        indices = partition.index[partition['partition']==train].tolist()
        self.attributes = pd.read_csv(os.path.join(root, attributes_csv)).iloc[indices]
        self.encoded_dir = encoded_dir
        
    def __len__(self):
        return len(self.attributes)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        encoding_name = os.path.join(self.encoded_dir, self.attributes.iloc[idx, 0]).replace('.jpg', '.npy')
        encoded_tensor = torch.from_numpy(np.load(encoding_name))
        attributes = self.attributes.iloc[idx, 1:].to_numpy(dtype=np.float32)
        attributes = torch.from_numpy(attributes)
        return encoded_tensor,attributes

class CelebAEncodedData(pl.LightningDataModule):
    def __init__(self, args):   
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.encoded_dir = os.path.join(args.encoded_dir, args.pretrained_ver)
    
    def get_dataloader(self, train):
        dataset = CelebAEncodedDataset(root=self.data_dir, train=train, encoded_dir=self.encoded_dir)
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

    def train_dataloader(self):
        return self.get_dataloader(0)

    def val_dataloader(self):
        return self.get_dataloader(1)

    def test_dataloader(self):
        return self.get_dataloader(2)


class CelebADataset(Dataset):
    def __init__(self, root, images_dir='img_align_celeba', partition_csv='list_eval_partition.csv', attributes_csv='list_attr_celeba.csv', train=0, transform=None):
        partition =  pd.read_csv(os.path.join(root, partition_csv))
        indices = partition.index[partition['partition']==train].tolist()
        self.attributes = pd.read_csv(os.path.join(root, attributes_csv)).iloc[indices]
        self.images_dir = os.path.join(root, images_dir)
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
        self.images_dir = args.images_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    def get_dataloader(self, train):
        transform = T.Compose([T.ToTensor()])
        dataset = CelebADataset(root=self.data_dir, images_dir=self.images_dir, train=train, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=(train==0),
            drop_last=True,
            persistent_workers=True
        )
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader(0)

    def val_dataloader(self):
        return self.get_dataloader(1)

    def test_dataloader(self):
        return self.get_dataloader(2)


class CheXpertDataset(Dataset):
    def __init__(self, root, transform, train):
        dirs = ['CheXpert-v1.0-small/train.csv', 'CheXpert-v1.0-small/valid.csv']
        self.attributes = pd.read_csv(os.path.join(root, dirs[train]))
        self.attributes = self.attributes[self.attributes['Frontal/Lateral']=="Frontal"]
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.attributes)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root, self.attributes.iloc[idx, 0])
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        return image, 1


class CheXpertData(pl.LightningDataModule):
    def __init__(self, args):   
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.num_workers = args.num_workers

    def get_dataloader(self, train):
        batch_sizes = [self.batch_size, self.val_batch_size]
        transform = T.Compose([T.ToTensor(),])
        dataset = CheXpertDataset(root=self.data_dir, train=train, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_sizes[train],
            num_workers=self.num_workers,
            shuffle=(train==0),
            drop_last=True,
            pin_memory=True,
            persistent_workers=True
        )
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader(0)

    def val_dataloader(self):
        return self.get_dataloader(1)




class CheXpertEncodedDataset(Dataset):
    def __init__(self, root, target_attr, train=0, transform=None):
        dirs = ['CheXpert-v1.0-small/train.csv', 'CheXpert-v1.0-small/valid.csv']
        self.attributes = pd.read_csv(os.path.join(root, dirs[train]))
        self.attributes = self.attributes[self.attributes['Frontal/Lateral']=="Frontal"].dropna(subset=[target_attr]).reset_index()
        self.root = root
        
    def __len__(self):
        return len(self.attributes)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        encoding_name = os.path.join(self.root, self.attributes.iloc[idx, 1]).replace('.jpg', '.npy')
        encoded_tensor = torch.from_numpy(np.load(encoding_name))
        attributes = self.attributes.iloc[idx, 6:].to_numpy(dtype=np.float32)
        attributes[attributes==-1]=1
        attributes[attributes==0]=-1
        attributes = torch.from_numpy(attributes)
        return encoded_tensor,attributes

class CheXpertEncodedData(pl.LightningDataModule):
    def __init__(self, args):   
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.num_workers = args.num_workers
        self.encoded_dir = os.path.join(args.encoded_dir, args.pretrained_ver)
        self.target_attr = args.target_attr
    
    def get_dataloader(self, train):
        dataset = CheXpertEncodedDataset(root=self.encoded_dir, target_attr=self.target_attr, train=train)
        dataloader = DataLoader(
            dataset,
            batch_size=[self.batch_size,self.val_batch_size][train],
            num_workers=self.num_workers,
            shuffle=(train==0),
            drop_last=True,
            pin_memory=True,
            persistent_workers=True
        )
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader(0)

    def val_dataloader(self):
        return self.get_dataloader(1)



class CheXpertCycleganDataset(Dataset):
    def __init__(self, root, train=0, target_attr='', transform=None):
        self.root = root
        dirs = ['CheXpert-v1.0-small/train.csv', 'CheXpert-v1.0-small/valid.csv']
        self.attributes = pd.read_csv(os.path.join(root, dirs[train]))
        self.attributes = self.attributes[self.attributes['Frontal/Lateral']=="Frontal"].dropna(subset=[target_attr]).reset_index()
        self.a_indices = self.attributes.index[self.attributes[target_attr]==1].tolist()
        self.b_indices = self.attributes.index[self.attributes[target_attr]==0].tolist()
        self.a_attributes = self.attributes.iloc[self.a_indices]
        self.b_attributes = self.attributes.iloc[self.b_indices]
        print(max(len(self.a_indices), len(self.b_indices)))
        
    def __len__(self):
        return max(len(self.a_indices), len(self.b_indices))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx_a = idx
        idx_b = idx
        if(len(self.a_indices)<len(self.b_indices)):
            idx_a = idx_a % len(self.a_indices)
        else:
            idx_b = idx_b % len(self.b_indices)
        
        # print(idx_a)
        # print(idx_b)
        encoded_a = torch.from_numpy(np.load(os.path.join(self.root, self.a_attributes.iloc[idx_a, 1]).replace('.jpg', '.npy')))
        attributes_a = torch.from_numpy(self.a_attributes.iloc[idx_a, 6:].to_numpy(dtype=np.float32))
        attributes_a[attributes_a==-1]=1

        encoded_b = torch.from_numpy(np.load(os.path.join(self.root, self.b_attributes.iloc[idx_b, 1]).replace('.jpg', '.npy')))
        attributes_b = torch.from_numpy(self.b_attributes.iloc[idx_b, 6:].to_numpy(dtype=np.float32))
        attributes_b[attributes_b==0]=-1
        
        return encoded_a, encoded_b, attributes_a, attributes_b

class CheXpertCycleganData(pl.LightningDataModule):
    def __init__(self, args):   
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.num_workers = args.num_workers
        self.target_attr = args.target_attr
        self.encoded_dir = os.path.join(args.encoded_dir, args.pretrained_ver)
    
    def get_dataloader(self, train):
        dataset = CheXpertCycleganDataset(root=self.encoded_dir, train=train, target_attr=self.target_attr)
        dataloader = DataLoader(
            dataset,
            batch_size=[self.batch_size,self.val_batch_size][train],
            num_workers=self.num_workers,
            shuffle=(train==0),
            drop_last=True,
            pin_memory=True,
            persistent_workers=True
        )
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader(0)

    def val_dataloader(self):
        return self.get_dataloader(1)


