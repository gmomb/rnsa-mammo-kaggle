import logging, datetime, os, argparse
from pathlib import Path
import pandas as pd
import torch
import neptune.new as neptune

from datasets import create_train_loader, create_valid_loader
from configs import cfg
from modeling.model import kaggleBCModel, kaggleNextVIT

from engine.fitter import Fitter
from utilities.utils import seed_everything

run = neptune.init_run(
    project="geno/rsna-mammo-kaggle",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMjA5NDA5Zi1kYTdhLTQ5MGMtOTU4Ni02ODU0YmJkNTdlNDQifQ==",
)

run['parameters'] = cfg

Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

#Istanzio il logger
path_logger = os.path.join(
    cfg.OUTPUT_DIR, f'train-{datetime.datetime.now()}.log'
)

logging.basicConfig(filename=path_logger, level=logging.DEBUG)
logger = logging.getLogger()

if __name__ == '__main__':

    torch.cuda.empty_cache()

    parser =  argparse.ArgumentParser(
        prog = 'KaggleRSNA',
    )
    parser.add_argument('--debug', default=False, type=bool)

    args = parser.parse_args()

    ##cfg.DEBUG = args.debug
    seed_everything(cfg.SEED)
    
    #LEggo il dataframe
    path_df = os.path.join(
        cfg.INPUT.ROOT_DIR, 'train_folds.csv'
    )

    df = pd.read_csv(
        path_df
    )

    train_df = df[df['fold'] != cfg.INPUT.VALID_FOLD]
    valid_df = df[df['fold'] == cfg.INPUT.VALID_FOLD]

    #Se sono in modalità debug itero velocemente
    if cfg.DEBUG:
        train_df = train_df.sample(frac=0.1)
        valid_df = valid_df.sample(frac=0.1)


    # Creo i loader
    train_loader = create_train_loader(cfg, df=train_df)
    valid_loader = create_valid_loader(cfg, df=valid_df)

    #Istanzio il modello
    model = kaggleNextVIT(cfg)

    #Istanzio il fitter
    engine = Fitter(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=valid_loader,
        logger=logger,
        neptune_runner = run
    )

    #Start the training
    logging.info(f'Started training with parameters: {cfg}')

    engine.fit()
    #engine.final_check()
    #engine.compute_shift()

    run.stop()