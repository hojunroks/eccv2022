from argparse import ArgumentParser
from src.models.ours import OurGan
from src.models.cyclegan import CycleGan
from src.models.autoencoder import AutoEncoder
from src.models.encodedclassifier import EncodedClassifier
from src.datamodule import CelebACycleganData
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from src.utils import parse_config
import sys
import os

def main():
    #######################
    # PARSE ARGUMENTS
    #######################
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument('--data_dir', type=str, required=False)
    parser.add_argument('--pretrained_dir', type=str, required=False)
    parser.add_argument('--pretrained_ver', type=str, required=False)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--num_workers", type=int, required=False)
    parser.add_argument("--target_attr", type=str, required=False)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    # add model specific args
    parser = OurGan.add_model_specific_args(parser)
    args = parse_config(parser, 'config.yml', 'translator')

    ###########################
    # INITIALIZE MODULES
    ###########################
    dm = CelebACycleganData(args)    
    autoencoder = AutoEncoder.load_from_checkpoint(hparams=args, checkpoint_path=os.path.join(args.pretrained_dir, args.pretrained_ver, args.pretrained_autoencoder))
    decoder = autoencoder.decoder.eval()
    translator = CycleGan.load_from_checkpoint(hparams=args, decoder=decoder, checkpoint_path = os.path.join(args.pretrained_dir, args.pretrained_ver, args.id_gen))
    classifier = EncodedClassifier.load_from_checkpoint(hparams=args, checkpoint_path=os.path.join(args.pretrained_dir, args.pretrained_ver, args.target_attr+".ckpt"))
    ours = OurGan(hparams=args, decoder=decoder, classifier=classifier, a2b= translator.A2B, b2a= translator.B2A)
    # translator = CycleGan(hparams=args, decoder=decoder)
    logger = TensorBoardLogger('logs/ourgan/{}'.format(datetime.now().strftime("/%m%d")), name='')

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
    
    trainer.fit(ours, datamodule=dm)
    

if __name__=='__main__':
    main()



