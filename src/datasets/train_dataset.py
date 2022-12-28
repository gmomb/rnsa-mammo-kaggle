import os, datetime

import cv2
from torch.utils.data import Dataset

class trainMammo(Dataset):
    
    def __init__(self, cfg, df, transforms = None) -> None:
        super().__init__()

        self.cfg = cfg
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx:int):

        row = self.df.iloc[idx]
        img_path = os.path.join(
            self.cfg.DATASETS.ROOT_DIR,
            'train_images_processed_cv2_256',
            str(row['patient_id']),
            str(row['image_id'])+".png"
        )

        img = cv2.imread(
            img_path
        )

        label = row['cancer']
        img_id = row['image_id']
        patient = row['patient_id']

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.transforms:
            augmented = self.transforms(image = img)
            img = augmented['image']

        return img, label, img_id, patient

