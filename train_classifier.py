from argparse import ArgumentParser
from src.models.encodedclassifier import EncodedClassifier
from src.datamodule import CelebAData
import pytorch_lightning as pl
from torchvision import models
from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger
from src.utils import parse_config
from src.datamodule import CelebAEncodedData
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

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
    parser.add_argument("--target_attr", type=int, required=False)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    # add model specific args
    parser = EncodedClassifier.add_model_specific_args(parser)
    args = parse_config(parser, 'config.yml', 'classifier')

    dm = CelebAEncodedData(args)
    classifier = EncodedClassifier(hparams=args, target_attr=args.target_attr)
    logger = TensorBoardLogger('logs/classifier/{}'.format(datetime.now().strftime("/%m%d")), name='')

    ###########################
    # TRAIN
    ###########################
    print("START TRAINING...")
    checkpoint_callback = ModelCheckpoint(monitor="loss/val")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer.from_argparse_args(args,
        logger=logger,
        callbacks = [checkpoint_callback, lr_monitor],  
    )
    
    trainer.fit(classifier, datamodule=dm)
    

    ###########################
    # TEST
    ###########################
    print("START TESTING...")
    # result = trainer.test(datamodule=dm)
    # print(result)

if __name__=='__main__':
    main()



