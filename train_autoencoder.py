from argparse import ArgumentParser
from src.autoencoder import AutoEncoder
from src.models.classifier import Classifier
from src.datamodule import CelebAData
import pytorch_lightning as pl
from torchvision import models
from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

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

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    # add model specific args
    parser = AutoEncoder.add_model_specific_args(parser)
    args = parser.parse_args()

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
    logdir += '/{}epochs'.format(args.max_epochs)
    logdir += '/{}'.format(args.optimizer)
    logger = TensorBoardLogger(logdir, name='')



    ###########################
    # TRAIN
    ###########################
    print("START TRAINING...")
    checkpoint_callback = ModelCheckpoint(monitor="loss/validation")
    trainer = pl.Trainer.from_argparse_args(args,
        logger=logger,
        fast_dev_run=False,
        callbacks = [checkpoint_callback],
        accumulate_grad_batches = 4,
    )
    
    trainer.fit(autoencoder, datamodule=dm)
    trainer.save_checkpoint(logger.log_dir+logger.name+"/"+logger.name+"celeba_test.ckpt")

    ###########################
    # TEST
    ###########################
    print("START TESTING...")

if __name__=='__main__':
    main()



