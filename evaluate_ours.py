import os
import torch
import matplotlib.pyplot as plt
from src.models.ours import OurGan
from argparse import ArgumentParser
from src.datamodule import CelebAEncodedDataset, CheXpertEncodedDataset
from torch.utils.data import DataLoader
from src.models.encodedclassifier import EncodedClassifier
from src.models.autoencoder import AutoEncoder
from src.utils import parse_config
import torch.nn as nn
from torchmetrics import Accuracy, AUROC, F1
import pandas as pd
import numpy as np
from math import exp
from sklearn.metrics import confusion_matrix
import random
ATTRIBUTE_KEYS = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
ATTRIBUTE_KEYS_CHEXPERT = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
       'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
       'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
       'Fracture', 'Support Devices']


def main():
    #######################
    # PARSE ARGUMENTS
    #######################
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument('--data_dir', type=str, required=False)
    parser.add_argument('--dataset', type=str, required=False)
    parser.add_argument('--pretrained_dir', type=str, required=False)
    parser.add_argument('--pretrained_ver', type=str, required=False)
    parser.add_argument('--use_pretrain', type=int, required=False)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--val_batch_size", type=int, required=False)
    parser.add_argument("--num_workers", type=int, required=False)
    parser.add_argument("--target_attr", type=str, required=False)
    parser.add_argument("--eps", type=float, required=False)


    parser = OurGan.add_model_specific_args(parser)
    args = parse_config(parser, 'config.yml', 'translator')
    root = os.path.join(args.encoded_dir, args.pretrained_ver)
    
    classifier = EncodedClassifier.load_from_checkpoint(hparams=args, checkpoint_path=os.path.join(args.pretrained_dir, args.pretrained_ver, args.target_attr+".ckpt"))
    classifiers = {}
    
    for cls in args.classifiers.split(','):
        print(cls)
        idx = ATTRIBUTE_KEYS.index(cls)
        classifiers[idx] = EncodedClassifier.load_from_checkpoint(hparams=args, checkpoint_path=os.path.join(args.pretrained_dir, args.pretrained_ver, cls+".ckpt"))    


    if args.dataset=='celeba':
        ds = CelebAEncodedDataset(root=args.data_dir, train=1, encoded_dir=root)
        target_attr_index = ATTRIBUTE_KEYS.index(args.target_attr)
        attribute_keys = ATTRIBUTE_KEYS
    else:
        ds = CheXpertEncodedDataset(root=root, train=1, target_attr=args.target_attr)
        target_attr_index = ATTRIBUTE_KEYS_CHEXPERT.index(args.target_attr)
        attribute_keys = ATTRIBUTE_KEYS_CHEXPERT

    autoencoder = AutoEncoder.load_from_checkpoint(hparams=args, checkpoint_path=os.path.join(args.pretrained_dir, args.pretrained_ver, args.pretrained_autoencoder))
    decoder = autoencoder.decoder.eval()
    for cls in args.classifiers.split(','):
        idx = attribute_keys.index(cls)
        classifiers[idx] = EncodedClassifier.load_from_checkpoint(hparams=args, checkpoint_path=os.path.join(args.pretrained_dir, args.pretrained_ver, cls+".ckpt")).to('cuda:0')
    
    ours = OurGan.load_from_checkpoint(hparams=args, decoder=decoder, classifier=classifier, classifiers=classifiers, checkpoint_path=os.path.join(args.pretrained_dir, "done", args.pretrained_ours+'.ckpt'))
    a2b = ours.A2B.eval().to('cuda:0')
    b2a = ours.B2A.eval().to('cuda:0')

    
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=0,
        drop_last=False        
    )


    true_attrs = pd.DataFrame()
    estimated_attrs = pd.DataFrame()
    translated_attrs = pd.DataFrame()
    
    for batch_index, batch in enumerate(dl):
        enc, attr = batch
        enc = enc.to('cuda:0')
        attr = ((attr+1)/2).long()
        true_attr = attr
        estimated_attr = torch.zeros(attr.shape)
        translated_attr = torch.zeros(attr.shape)

        a_indices = (attr[:, target_attr_index] == 1).nonzero(as_tuple=True)[0]
        b_indices = (attr[:, target_attr_index] == 0).nonzero(as_tuple=True)[0]
        a_translate = torch.rand(a_indices.shape)
        b_translate = torch.rand(b_indices.shape)
        fakeb_indices = a_indices[(a_translate < (1/exp(args.eps))).nonzero(as_tuple=True)[0]]
        fakea_indices = b_indices[(b_translate < (1/exp(args.eps))).nonzero(as_tuple=True)[0]]
        
        with torch.no_grad():
            for cls in classifiers.keys():
                classifiers[cls].eval()
                estimated_attr[:, cls]= torch.argmax(classifiers[cls](enc), 1).cpu()

            enc[fakea_indices] = torch.flatten(b2a(nn.Unflatten(1, (512, 4, 4))(enc[fakea_indices])), start_dim=1)
            enc[fakeb_indices] = torch.flatten(a2b(nn.Unflatten(1, (512, 4, 4))(enc[fakeb_indices])), start_dim=1)

            for cls in classifiers.keys():
                classifiers[cls].to('cuda:0')
                classifiers[cls].eval()
                translated_attr[:, cls]= torch.argmax(classifiers[cls](enc), 1).cpu()

        c = list(classifiers.keys())
        pd.concat((true_attrs, true_attr[c]))
        pd.concat((estimated_attrs, estimated_attr[c]))
        pd.concat((translated_attrs, translated_attr[c]))


    for i in range(len(args.classifiers)):
        print("Confusion matrix of true vs. pred of {} for estimated, true: {}").format(args.classifiers[i], confusion_matrix(true_attrs[i], estimated_attrs[i]))
        
    
    print(true_attrs.count())
    print(estimated_attrs.count())
    print(translated_attrs.count())
    
        


        
        





if __name__=='__main__':
    main()



