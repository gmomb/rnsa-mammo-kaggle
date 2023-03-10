# encoding: utf-8
import torch

def make_optimizer(cfg, model):

    #nbs = cfg.SOLVER.NOMINAL_BATCH_SIZE #nominal batch size
    #bs = cfg.SOLVER.IMS_PER_BATCH #batch size

    #TODO: gradient accumulation?

    #params = []
    #for key, value in model.named_parameters():
    #    if not value.requires_grad:
    #        continue
    #    lr = cfg.SOLVER.BASE_LR
    #    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    #    if "bias" in key:
    #        lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
    #        weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
    #    if "bn" in key:
    #        weight_decay = cfg.SOLVER.WEIGHT_DECAY_BN
    #    
    #    params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    
    #TODO: aggiungere lr decay
    params = [{
        "params": model.parameters(), 
        "lr": cfg.SOLVER.BASE_LR}
    ]
    #fix del momentum
    if cfg.SOLVER.OPTIMIZER_NAME == "SGD":
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM, nesterov=True)
    elif cfg.SOLVER.OPTIMIZER_NAME == "Adam":
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    elif cfg.SOLVER.OPTIMIZER_NAME == "AdamW":
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, weight_decay = cfg.SOLVER.WEIGHT_DECAY)
    else:
        print('Attenzione, ottimizzatore non riconosciuto!')
    return optimizer
