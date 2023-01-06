import os

from torch.utils.data import DataLoader

from .train_dataset import trainMammo
from .transforms import get_train_transform, get_valid_transform

def create_train_loader(cfg, df):

    #create training transformation
    train_trnsf = get_train_transform(
        cfg=cfg
    )

    # create dataset for training
    train_dataset = trainMammo(
        cfg = cfg,
        df = df,
        transforms=train_trnsf
    )

    #build training iterator
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.INPUT.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=True
    )

    return train_loader

def create_valid_loader(cfg, df):

    #create valid transformation
    valid_trnsf = get_valid_transform(
        cfg=cfg
    )

    # create dataset for validation
    valid_dataset = trainMammo(
        cfg = cfg,
        df = df,
        transforms=valid_trnsf
    )

    #build valid iterator
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.INPUT.TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )

    return valid_loader
