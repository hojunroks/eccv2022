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

def main():
    #######################
    # PARSE ARGUMENTS
    #######################
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument('--data_dir', type=str, required=False)
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
    autoencoder = AutoEncoder.load_from_checkpoint(hparams=args, checkpoint_path=args.pretrained_autoencoder)
    decoder = autoencoder.decoder.eval()
    if args.pretrain:
        translator = CycleGan(hparams=args, decoder=decoder)
    else:
        translator = CycleGan.load_from_checkpoint(hparams=args, decoder=decoder, checkpoint_path = args.id_gen)
    # translator = CycleGan(hparams=args, decoder=decoder)
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
    )
    
    trainer.fit(translator, datamodule=dm)
    

if __name__=='__main__':
    main()



