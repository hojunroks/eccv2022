from argparse import ArgumentParser
from src.models.cyclegan import CycleGan
from src.datamodule import CelebAEncodedData
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger
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

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    # add model specific args
    parser = CycleGan.add_model_specific_args(parser)
    args = parse_config(parser, 'config.yml', 'translator')

    ###########################
    # INITIALIZE MODULES
    ###########################
    dm = CelebAEncodedData(args)    
    translator = CycleGan(args, target_attr=21)
    logger = TensorBoardLogger('logs/{}'.format(datetime.now().strftime("/%m%d")), name='')

    ###########################
    # TRAIN
    ###########################
    print("START TRAINING...")
    # checkpoint_callback = ModelCheckpoint(monitor="loss/validation")
    trainer = pl.Trainer.from_argparse_args(args,
        logger=logger,
        # callbacks = [checkpoint_callback],
        accumulate_grad_batches = 4,
        gradient_clip_val=0.5
    )
    
    trainer.fit(translator, datamodule=dm)
    
    trainer.save_checkpoint(logger.log_dir+logger.name+"/"+logger.name+"celeba_test.ckpt")

    ###########################
    # TEST
    ###########################
    print("START TESTING...")
    # result = trainer.test(datamodule=dm)
    # print(result)

if __name__=='__main__':
    main()



