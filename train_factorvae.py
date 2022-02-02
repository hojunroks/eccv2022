from argparse import ArgumentParser
from src.models.factorvae import FactorVAE
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

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    # add model specific args
    parser = FactorVAE.add_model_specific_args(parser)
    args = parse_config(parser, 'config.yml', 'fvae')

    ###########################
    # INITIALIZE DATAMODULE
    ###########################
    print("INITIALIZING DATAMODULE...")
    dm = CelebAData(args)
    
    ###########################
    # INITIALIZE MODEL
    ###########################
    print("INITIALIZING MODEL...")
    fvae = FactorVAE(args)

    ###########################
    # INITIALIZE LOGGER
    ###########################
    print("INITIALIZING LOGGER...")
    logdir = 'logs/factorvae'
    logdir += datetime.now().strftime("/%m%d")
    logger = TensorBoardLogger(logdir, name='')



    ###########################
    # TRAIN
    ###########################
    print("START TRAINING...")
    checkpoint_callback = ModelCheckpoint(monitor="vae/loss")
    trainer = pl.Trainer.from_argparse_args(args,
        logger=logger,
        fast_dev_run=False,
        callbacks = [checkpoint_callback],
        accumulate_grad_batches = 1,
    )
    
    trainer.fit(fvae, datamodule=dm)
    trainer.save_checkpoint(logger.log_dir+logger.name+"/"+logger.name+"fvae.ckpt")

    ###########################
    # TEST
    ###########################
    print("START TESTING...")

if __name__=='__main__':
    main()



