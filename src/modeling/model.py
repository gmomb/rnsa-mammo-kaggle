import os

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from timm import create_model

from .next_vit import nextvit_base

class kaggleBCModel(torch.nn.Module):
    def __init__(self, aux_class, cfg) -> None:
        super().__init__()

        self.cfg = cfg
        #TODO: ricordarsi nella prediction di fare un sigmoid dopo l'output
        self.register_buffer('mean', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
        self.encoder = create_model(
            self.cfg.SOLVER.MODEL_NAME, pretrained=True, in_chans = 3
        )

        self.fe_dim = self.encoder.fc.in_features
        
        self.cancer_layer = torch.nn.Sequential(
            torch.nn.Linear(self.fe_dim, 1),
        )

        self.aux_layer = torch.nn.ModuleList([
            torch.nn.Linear(self.fe_dim, aux_dim) for aux_dim in aux_class
        ])

    def forward(self, x):

        x = (x - self.mean) / self.std

        e = self.encoder.forward_features(x)
        x = F.adaptive_avg_pool2d(e, 1)
        x = torch.flatten(x,1,3)
        cancer = self.cancer_layer(x).reshape(-1)

        aux_pred = []
        for layer in self.aux_layer:
            aux_pred.append(layer(x))

        return cancer, aux_pred

    def predict(self, x):
    
        #return sigmod/softmax
        cancer_logits, aux_logits = self.forward(x)

        x_aux = []
        for layer in aux_logits:
            x_aux.append(torch.softmax(layer, dim=1))

        return torch.sigmoid(cancer_logits), x_aux


class kaggleNextVIT(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg
        #TODO: ricordarsi nella prediction di fare un sigmoid dopo l'output
        #inchans=1 per avere solo un layer nell'immagine
        self.encoder = nextvit_base()
        if cfg.SOLVER.PRETRAINED:
            checkpoint = torch.load(self.cfg.SOLVER.PRETRAINED_PATH)
            self.encoder.load_state_dict(checkpoint['model'])

        self.encoder.proj_head = torch.nn.Linear(in_features=1024, out_features=1)
            
    def forward(self, x):

        x = self.encoder(x)

        return x

#https://github.com/rwightman/pytorch-image-models/blob/main/timm/loss/cross_entropy.py
class LabelSmoothingCrossEntropy(torch.nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

#https://gist.github.com/MrRobot2211/efc2aec30cb9323b7664ef1702846c08
class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, pos_weight=None, reduction='mean', smoothing=0.0):
        super().__init__(reduction=reduction)
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss