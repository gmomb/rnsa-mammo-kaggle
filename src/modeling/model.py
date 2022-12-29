import os

import torch
from timm import create_model

class kaggleBCModel(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg
        #TODO: ricordarsi nella prediction di fare un sigmoid dopo l'output
        #inchans=1 per avere solo un layer nell'immagine
        self.model = create_model(self.cfg.SOLVER.MODEL_NAME, pretrained=True, num_classes=0, in_chans = 1)

        self.fe_dim = list(self.model.parameters())[-1].shape[0]
        self.fc_output = torch.nn.Sequential(
            torch.nn.Linear(self.fe_dim, 1),
        )

    def forward(self, x):

        x = self.model(x)
        preds = self.fc_output(x)

        return preds