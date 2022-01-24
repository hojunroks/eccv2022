from argparse import ArgumentParser
from src.ppnet import PPNet
from src.datamodule import CelebAEncodedData
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger

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
    parser = PPNet.add_model_specific_args(parser)
    args = parser.parse_args()

    ###########################
    # INITIALIZE DATAMODULE
    ###########################
    print("INITIALIZING DATAMODULE...")
    dm = CelebAEncodedData(args)
    
    ###########################
    # INITIALIZE MODEL
    ###########################
    print("INITIALIZING MODEL...")
    args.target_attr = 21
    ppnet = PPNet(args, target_attr=21)

    ###########################
    # INITIALIZE LOGGER
    ###########################
    print("INITIALIZING LOGGER...")
    logdir = 'logs/ppnet'
    logdir += datetime.now().strftime("/%m%d")
    logdir += '/{}epochs'.format(args.max_epochs)
    logger = TensorBoardLogger(logdir, name='')

    ###########################
    # TRAIN
    ###########################
    print("START TRAINING...")
    # checkpoint_callback = ModelCheckpoint(monitor="loss/validation")
    trainer = pl.Trainer.from_argparse_args(args,
        logger=logger,
        fast_dev_run=False,
        deterministic=True,
        enable_model_summary=False,
        log_every_n_steps=1,
        # callbacks = [checkpoint_callback],
        accumulate_grad_batches = 4,
        gradient_clip_val=0.5
    )
    
    trainer.fit(ppnet, datamodule=dm)
    
    trainer.save_checkpoint(logger.log_dir+logger.name+"/"+logger.name+"celeba_test.ckpt")

    ###########################
    # TEST
    ###########################
    print("START TESTING...")
    # result = trainer.test(datamodule=dm)
    # print(result)

if __name__=='__main__':
    main()



