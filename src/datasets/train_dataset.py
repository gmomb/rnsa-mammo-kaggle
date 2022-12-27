import os, datetime

from torch.utils.data import Dataset

class trainMammo(Dataset):
    
    def __init__(self, cfg, img_ids, transforms = None) -> None:
        super().__init__()


    