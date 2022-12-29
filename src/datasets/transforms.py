import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(cfg):

    return A.Compose([
        #TODO: improve transformations
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