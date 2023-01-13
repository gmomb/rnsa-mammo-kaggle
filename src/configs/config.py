# encoding: utf-8
import os, datetime

from pathlib import Path
from yacs.config import CfgNode as CN

import torch

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.DEBUG = False
_C.SEED = 1234
_C.VERBOSE = True
_C.PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
_C.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

_C.MODEL = CN()
_C.MODEL.NUM_CLASSES = 2

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.IMG_SIZE = 256
_C.INPUT.ROOT_DIR = os.path.join(_C.PROJECT_ROOT, 'data')
# Fold to validate
_C.INPUT.VALID_FOLD = 0
# # List of the dataset names for training, as present in paths_catalog.py
# _C.DATASETS.TRAIN = ()
# # List of the dataset names for testing, as present in paths_catalog.py
# _C.DATASETS.TEST = ()
# Stratifico
_C.INPUT.STRATIFIED = True
_C.INPUT.N_SPLITS = 5
_C.INPUT.TRAIN_BATCH_SIZE = 32
# RandomSizedCrop paramters
_C.INPUT.RSC_HEIGHT = _C.INPUT.IMG_SIZE
_C.INPUT.RSC_WIDTH = _C.INPUT.IMG_SIZE
_C.INPUT.RSC_SCALE = (0.8, 1)
_C.INPUT.RSC_RATIO = (0.45, 0.55)
_C.INPUT.RSC_PROB = 0.5
# Coutout paramters
_C.INPUT.COTOUT_NUM_HOLES = 8
_C.INPUT.COTOUT_MAX_H_SIZE = 64
_C.INPUT.COTOUT_MAX_W_SIZE = 64
_C.INPUT.COTOUT_FILL_VALUE = 0
_C.INPUT.COTOUT_PROB = 0.4


# Random HorizontalFlip
_C.INPUT.HFLIP_PROB = 0.5
#Random rotate
_C.INPUT.ROTATE_PROB = 0.5

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MODEL_NAME = 'seresnext50_32x4d'
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.NOMINAL_BATCH_SIZE = 50
_C.SOLVER.SCHEDULER_NAME = "LambdaLR"
#_C.SOLVER.SCHEDULER_NAME = "LambdaLR"
_C.SOLVER.TARGET_SMOOTHING = 0.1
_C.SOLVER.MAX_EPOCHS = 5
_C.SOLVER.BASE_LR = 1e-4
_C.SOLVER.POS_TARGET_WEIGHT = 20
_C.SOLVER.LR_MULT = 0.9
_C.SOLVER.WEIGHT_DECAY = 1e-2
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 4
_C.TEST.WEIGHT = "/output/best-checkpoint.bin"

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = str(os.path.join(
    _C.PROJECT_ROOT, 
    'experiments', 
    _C.SOLVER.MODEL_NAME+'_'+str(datetime.date.today()), 
))

_C.WEIGHT_PATH = str(os.path.join(
    _C.OUTPUT_DIR, 'weights', _C.SOLVER.MODEL_NAME+'.path'
))