# encoding: utf-8
import torch
import math

def make_scheduler(cfg, optimizer, train_loader):
    #number_of_iteration_per_epoch = len(train_loader)
    #learning_rate_step_size = cfg.SOLVER.COS_CPOCH * number_of_iteration_per_epoch
    #lf = lambda x: (((1 + math.cos(x * math.pi / cfg.SOLVER.MAX_EPOCHS)) / 2) ** 1.0) * 0.9 + 0.1
    #scheduler = getattr(torch.optim.lr_scheduler, cfg.SOLVER.SCHEDULER_NAME)(optimizer, T_0=learning_rate_step_size, T_mult=cfg.SOLVER.T_MUL)
    lambda_lr = lambda epoch: cfg.SOLVER.LR_MULT ** epoch
    scheduler = getattr(torch.optim.lr_scheduler, cfg.SOLVER.SCHEDULER_NAME)(optimizer, lr_lambda=lambda_lr)
    return scheduler