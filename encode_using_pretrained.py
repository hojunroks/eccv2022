from argparse import ArgumentParser
from src.models.autoencoder import AutoEncoder
import pandas as pd
from src.utils import parse_config
import os
import numpy as np
import torch
from skimage import io
from torchvision import transforms as T

def main():
    print("START PROGRAM")
    #######################
    # PARSE ARGUMENTS
    #######################
    print("PARSING ARGUMENTS...")
    parser = ArgumentParser()    
    # add PROGRAM level args
    parser.add_argument('--data_dir', type=str, required=False)

    # add model specific args
    parser = AutoEncoder.add_model_specific_args(parser)
    args = parse_config(parser, 'config.yml', 'autoencoder')

    attributes_csv='list_attr_celeba.csv'
    attributes = pd.read_csv(os.path.join(args.data_dir, attributes_csv))
    images_dir = os.path.join(args.data_dir, args.images_dir)
    encoded_dir = os.path.join(args.encoded_dir, args.pretrained_ver)

    
    print("LOADING PRETRAINED MODEL...")
    autoencoder = AutoEncoder.load_from_checkpoint(hparams=args, checkpoint_path=os.path.join(args.pretrained_dir, args.pretrained_ver, args.pretrained_autoencoder)).to('cuda:0')
    encoder = autoencoder.encoder.eval()

    ###########################
    # ENCODE
    ###########################
    print("START ENCODING...")
    with torch.no_grad():
        for i in range(attributes['image_id'].count()):
            img = attributes['image_id'][i]
            t = torch.unsqueeze(T.ToTensor()(io.imread(os.path.join(images_dir, img))), 0).to('cuda:0')
            encoded = encoder(t).squeeze().flatten().cpu().numpy()
            np.save(os.path.join(encoded_dir, img.replace('.jpg', '.npy')), encoded)
            if i%1000==0:
                print(i)
        
        
    

    

if __name__=='__main__':
    main()



