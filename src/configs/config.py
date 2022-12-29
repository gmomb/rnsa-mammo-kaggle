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

_C.DEBUG = True
_C.SEED = 1234
_C.VERBOSE = True
_C.PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
_C.DEVICE = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

_C.MODEL = CN()
_C.MODEL.NUM_CLASSES = 2

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.IMG_SIZE = 64
_C.INPUT.ROOT_DIR = os.path.join(_C.PROJECT_ROOT, 'data')
# Fold to validate
_C.INPUT.VALID_FOLD = 1
# # List of the dataset names for training, as present in paths_catalog.py
# _C.DATASETS.TRAIN = ()
# # List of the dataset names for testing, as present in paths_catalog.py
# _C.DATASETS.TEST = ()
# Stratifico
_C.INPUT.STRATIFIED = True
_C.INPUT.N_SPLITS = 5
_C.INPUT.TRAIN_BATCH_SIZE = 8
# RandomSizedCrop paramters
_C.INPUT.RSC_MIN_MAX_HEIGHT = (int(_C.INPUT.IMG_SIZE*0.7), int(_C.INPUT.IMG_SIZE*0.7))
_C.INPUT.RSC_HEIGHT = _C.INPUT.IMG_SIZE
_C.INPUT.RSC_WIDTH = _C.INPUT.IMG_SIZE
_C.INPUT.RSC_PROB = 0.5
# HueSaturationValue paramters
_C.INPUT.HSV_H = 0.2
_C.INPUT.HSV_S = 0.2
_C.INPUT.HSV_V = 0.2
_C.INPUT.HSV_PROB = 0.9
# RandomBrightnessContrast paramters
_C.INPUT.BC_B = 0.2
_C.INPUT.BC_C = 0.2
_C.INPUT.BC_PROB = 0.9
# Color paramters
_C.INPUT.COLOR_PROB = 0.9
# Random probability for ToGray
_C.INPUT.TOFGRAY_PROB = 0.01
# Random probability for HorizontalFlip
_C.INPUT.HFLIP_PROB = 0.5
# Random probability for VerticalFlip
_C.INPUT.VFLIP_PROB = 0.5
# Coutout paramters
_C.INPUT.COTOUT_NUM_HOLES = 8
_C.INPUT.COTOUT_MAX_H_SIZE = 64
_C.INPUT.COTOUT_MAX_W_SIZE = 64
_C.INPUT.COTOUT_FILL_VALUE = 0
_C.INPUT.COTOUT_PROB = 0.4

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 2

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MODEL_NAME = 'resnet34' #seresnext50_32x4d
_C.SOLVER.OPTIMIZER_NAME = "SGD"
_C.SOLVER.NOMINAL_BATCH_SIZE = 64
_C.SOLVER.SCHEDULER_NAME = "CosineAnnealingWarmRestarts"
#_C.SOLVER.SCHEDULER_NAME = "LambdaLR"
_C.SOLVER.COS_CPOCH = 2
_C.SOLVER.T_MUL = 2

_C.SOLVER.MAX_EPOCHS = 60

_C.SOLVER.BASE_LR = 1e-3
_C.SOLVER.BIAS_LR_FACTOR = 1

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0
_C.SOLVER.WEIGHT_DECAY_BN = 0

_C.SOLVER.WARMUP_EPOCHS = 10

_C.SOLVER.EARLY_STOP_PATIENCE = 20

_C.SOLVER.TRAIN_CHECKPOINT = False
_C.SOLVER.CLEAR_OUTPUT = True

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