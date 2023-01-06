import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(cfg):

    return A.Compose([
        #TODO: improve transformations
        A.HorizontalFlip(p=cfg.INPUT.HFLIP_PROB),
        A.RandomRotate90(p=cfg.INPUT.ROTATE_PROB),
        A.RandomResizedCrop(height = cfg.INPUT.RSC_HEIGHT, width = cfg.INPUT.RSC_WIDTH, scale=cfg.INPUT.RSC_SCALE, ratio=cfg.INPUT.RSC_RATIO, p=cfg.INPUT.RSC_PROB),
        A.Resize(height=cfg.INPUT.IMG_SIZE, width=cfg.INPUT.IMG_SIZE),
        ToTensorV2()
    ])

def get_valid_transform(cfg):
    return A.Compose([
        A.Resize(height=cfg.INPUT.IMG_SIZE, width=cfg.INPUT.IMG_SIZE),
        ToTensorV2()
    ])

def get_test_transform(cfg):
    return A.Compose([
        A.Resize(height=cfg.INPUT.IMG_SIZE, width=cfg.INPUT.IMG_SIZE),
        ToTensorV2()
    ])