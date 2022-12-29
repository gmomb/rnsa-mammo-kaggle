import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from configs import cfg

if __name__ == '__main__':
    
    path_file = os.path.join(
        cfg.INPUT.ROOT_DIR, 'train.csv'
    )

    df = pd.read_csv(path_file)

    df['fold'] = -1
    
    if cfg.INPUT.STRATIFIED:
        # stratifico i fold per il target
        kf = StratifiedKFold(
            n_splits=cfg.INPUT.N_SPLITS, 
            shuffle=True,
            random_state=cfg.SEED
        )

        for idx, (train_idx, val_idx) in enumerate(kf.split(X=df.index , y=df['cancer'])):
            print('Creating fold {}: {}'.format(idx, len(val_idx)))
            df.loc[val_idx, 'fold'] = idx

    else:
        # senza stratificazione
        kf = KFold(
            n_splits=cfg.INPUT.N_SPLITS, 
            shuffle=True, 
            random_state=cfg.SEED
        )

        for idx, (train_idx, val_idx) in enumerate(kf.split(X=df.index)):
            print('Creating fold {}: {}'.format(idx, len(val_idx)))
            df.loc[val_idx, 'fold'] = idx

    df.to_csv(
        os.path.join(
            cfg.INPUT.ROOT_DIR,
            'train_folds.csv'
        )
    )