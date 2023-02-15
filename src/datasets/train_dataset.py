import os

import numpy as np

import cv2
from torch.utils.data import Dataset, Sampler

def to_list(x):
    if isinstance(x, list): return x
    if isinstance(x, str): return eval(x)

class trainMammo(Dataset):
    def __init__(self, cfg, df, transforms = None):
        self.df = df
        self.cfg = cfg
        self.transform = transforms
        self.length = len(df)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]

        image = np.zeros(
            (self.cfg.INPUT.IMG_HEIGHT,self.cfg.INPUT.IMG_WIDTH), 
            np.uint8
        )

        img_path = os.path.join(
            self.cfg.INPUT.ROOT_DIR,
            'bc_1280_train_lut',
            f"{str(d['patient_id'])}_{str(d['image_id'])}.png"
        )

        m = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        h, w = m.shape

        try: #for degenerate case
            xmin, ymin, xmax, ymax = (np.array(to_list(d.pad_breast_box)) * h).astype(int)
            crop = m[ymin:ymax, xmin:xmax]

            mh, mw = (np.array(to_list(d.max_pad_breast_shape)) * h).astype(int)
            scale = min(self.cfg.INPUT.IMG_HEIGHT/mh,  self.cfg.INPUT.IMG_WIDTH/mw)
            dsize = (min(self.cfg.INPUT.IMG_WIDTH, int(scale * crop.shape[1])), min(self.cfg.INPUT.IMG_HEIGHT, int(scale * crop.shape[0])))
            if dsize != (crop.shape[1], crop.shape[0]):
                crop = cv2.resize(crop, dsize=dsize, interpolation=cv2.INTER_LINEAR)
            ch,cw = crop.shape  
            x = (self.cfg.INPUT.IMG_WIDTH  - cw) // 2
            y = (self.cfg.INPUT.IMG_HEIGHT - ch) // 2
            image[y:y + ch, x:x + cw] = crop

        except:
            crop  = m
            scale = min(self.cfg.INPUT.IMG_HEIGHT / h, self.cfg.INPUT.IMG_WIDTH / w)
            dsize = (min(self.cfg.INPUT.IMG_WIDTH, int(scale * crop.shape[1])), min(self.cfg.INPUT.IMG_HEIGHT, int(scale * crop.shape[0])))
            if dsize != (crop.shape[1], crop.shape[0]):
                crop = cv2.resize(crop, dsize=dsize, interpolation=cv2.INTER_LINEAR)
            ch, cw = crop.shape
            x = (self.cfg.INPUT.IMG_WIDTH  - cw) // 2
            y = (self.cfg.INPUT.IMG_HEIGHT - ch) // 2
            image[y:y + ch, x:x + cw] = crop
            
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            image = image.unsqueeze(0) 
        
        label = d['cancer']
        img_meta = {
            'img_id': d['image_id'],
            'patient_id': d['patient_id'],
            'laterality': d['laterality'],
            'site_id': d['site_id']
        }

        aux_target = d[self.cfg.INPUT.AUX_TARGETS].to_numpy(dtype=int)
        
        #image CHW con transforms, HWC altrimenti
        return image, label, aux_target, img_meta


#Per avere 1 esempio positivo su ratio
class BalanceSampler(Sampler):

    def __init__(self, dataset, ratio=8):
        self.r = ratio-1
        self.dataset = dataset
        self.pos_index = np.where(dataset["cancer"]>0)[0]
        self.neg_index = np.where(dataset["cancer"]==0)[0]

        N = int(np.floor(len(self.neg_index)/self.r))
        self.neg_length = self.r*N
        self.pos_length = N
        self.length = self.neg_length + self.pos_length


    def __iter__(self):
        pos_index = self.pos_index.copy()
        neg_index = self.neg_index.copy()
        np.random.shuffle(pos_index)
        np.random.shuffle(neg_index)

        neg_index = neg_index[:self.neg_length].reshape(-1,self.r)
        pos_index = np.random.choice(pos_index, self.pos_length).reshape(-1,1)
        index = np.concatenate([pos_index,neg_index],-1).reshape(-1)
        return iter(index)

    def __len__(self):
        return self.length