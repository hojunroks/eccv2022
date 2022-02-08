from argparse import ArgumentParser
from src.models.autoencoder import AutoEncoder
from src.datamodule import CelebAData
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.utils import parse_config

def main():
    print("START PROGRAM")
    #######################
    # PARSE ARGUMENTS
    #######################
    print("PARSING ARGUMENTS...")
    parser = ArgumentParser()
    
    # add PROGRAM level args
    parser.add_argument('--data_dir', type=str, required=False)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--num_workers", type=int, required=False)

    # add trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # add model specific args
    parser = AutoEncoder.add_model_specific_args(parser)
    args = parse_config(parser, 'config.yml', 'autoencoder')

    ###########################
    # INITIALIZE DATAMODULE
    ###########################
    print("INITIALIZING DATAMODULE...")
    dm = CelebAData(args)
    
    ###########################
    # INITIALIZE MODEL
    ###########################
    print("INITIALIZING MODEL...")
    autoencoder = AutoEncoder(args)

    ###########################
    # INITIALIZE LOGGER
    ###########################
    print("INITIALIZING LOGGER...")
    logdir = 'logs/autoencoder'
    logdir += datetime.now().strftime("/%m%d")
    logger = TensorBoardLogger(logdir, name='')

    ###########################
    # TRAIN
    ###########################
    print("START TRAINING...")
    checkpoint_callback = ModelCheckpoint(monitor="loss/validation")
    trainer = pl.Trainer.from_argparse_args(args,
        logger=logger,
        callbacks = [checkpoint_callback],
    )
    trainer.fit(autoencoder, datamodule=dm)
    trainer.save_checkpoint(logger.log_dir+logger.name+"/"+logger.name+"celeba_test.ckpt")

    ###########################
    # TEST
    ###########################
    print("START TESTING...")

if __name__=='__main__':
    main()



