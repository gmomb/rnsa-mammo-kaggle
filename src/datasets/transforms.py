import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(cfg):

    return A.Compose([
        #TODO: improve transformations
        A.HorizontalFlip(p=cfg.INPUT.HFLIP_PROB),
        A.RandomRotate90(p=cfg.INPUT.ROTATE_PROB),
        A.Cutout(
            num_holes=cfg.INPUT.COTOUT_NUM_HOLES,
            max_h_size=cfg.INPUT.COTOUT_MAX_H_SIZE,
            max_w_size=cfg.INPUT.COTOUT_MAX_W_SIZE,
            fill_value=cfg.INPUT.COTOUT_FILL_VALUE,
            p=cfg.INPUT.COTOUT_PROB
        ),
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