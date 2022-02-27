from argparse import ArgumentParser
from src.models.cyclegan import CycleGan
from src.models.autoencoder import AutoEncoder
from src.datamodule import CelebACycleganData
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from src.utils import parse_config
import sys
import os
from src.models.encodedclassifier import EncodedClassifier
ATTRIBUTE_KEYS = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

def main():
    #######################
    # PARSE ARGUMENTS
    #######################
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument('--data_dir', type=str, required=False)
    parser.add_argument('--pretrained_dir', type=str, required=False)
    parser.add_argument('--pretrained_ver', type=str, required=False)
    parser.add_argument('--use_pretrain', type=int, required=False)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--num_workers", type=int, required=False)
    parser.add_argument("--target_attr", type=str, required=False)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    # add model specific args
    parser = CycleGan.add_model_specific_args(parser)
    args = parse_config(parser, 'config.yml', 'translator')

    ###########################
    # INITIALIZE MODULES
    ###########################
    dm = CelebACycleganData(args)    
    autoencoder = AutoEncoder.load_from_checkpoint(hparams=args, checkpoint_path=os.path.join(args.pretrained_dir, args.pretrained_ver, args.pretrained_autoencoder))
    decoder = autoencoder.decoder.eval()
    classifier = EncodedClassifier.load_from_checkpoint(hparams=args, checkpoint_path=os.path.join(args.pretrained_dir, args.pretrained_ver, args.target_attr+".ckpt"))
    classifiers = {}
    for cls in args.classifiers.split():
        print(cls)
        idx = ATTRIBUTE_KEYS.index(cls)
        classifiers[idx] = EncodedClassifier.load_from_checkpoint(hparams=args, checkpoint_path=os.path.join(args.pretrained_dir, args.pretrained_ver, cls+".ckpt"))
    


    translator = CycleGan(hparams=args, decoder=decoder, classifiers=classifiers)
    logger = TensorBoardLogger('logs/cgan/{}'.format(datetime.now().strftime("/%m%d")), name='')

    ###########################
    # TRAIN
    ###########################
    print("START TRAINING...")
    # checkpoint_callback = ModelCheckpoint(monitor="loss/validation")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer.from_argparse_args(args,
        logger=logger,
        callbacks = [lr_monitor],  
        limit_train_batches = args.limit_train_batches, 
        limit_val_batches = args.limit_val_batches,
    )
    
    
    trainer.fit(translator, datamodule=dm)
    

if __name__=='__main__':
    main()



