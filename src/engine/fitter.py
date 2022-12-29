import os
import time
import sys
import warnings
from datetime import datetime
import torch
from torch.cuda.amp import GradScaler
from .average import AverageMeter
from evaluate.inference import inference
from evaluate.evaluate import evaluate
from tqdm import tqdm
import pandas as pd
from modeling.optimizer import make_optimizer
from modeling.scheduler import make_scheduler

import logging
warnings.filterwarnings("ignore")

class Fitter:
    def __init__(self, model, cfg, train_loader, val_loader, logger):
        self.config = cfg
        self.epoch = 0
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.base_dir = f'{self.config.OUTPUT_DIR}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.logger = logger
        self.best_final_score = 0.0
        self.best_score_threshold = 0.5

        self.model = model
        self.device = self.cfg.DEVICE
        self.model.to(self.device)

        self.optimizer = make_optimizer(cfg, model)
        self.scheduler = make_scheduler(cfg, self.optimizer, train_loader)
        self.scaler = GradScaler()

        self.logger.info(f'Fitter prepared. Device is {self.device}')
        self.all_predictions = []
        #self.early_stop_epochs = 0

        #self.early_stop_patience = self.config.SOLVER.EARLY_STOP_PATIENCE
        self.do_scheduler = False
        self.logger.info("Start training")

    def fit(self):
        for epoch in range(self.epoch, self.config.SOLVER.MAX_EPOCHS ):
            
            #TODO: reinserire warmup epochs
            #if epoch < self.config.SOLVER.WARMUP_EPOCHS:
            #    lr_scale = min(1., float(epoch + 1) / float(self.config.SOLVER.WARMUP_EPOCHS))
            #    for pg in self.optimizer.param_groups:
            #        pg['lr'] = lr_scale * self.config.SOLVER.BASE_LR
            #    self.do_scheduler = False
            #else:
            #    self.do_scheduler = True
            #if self.config.VERBOSE:
            #    lr = self.optimizer.param_groups[0]['lr']
            #    timestamp = datetime.utcnow().isoformat()
            #    self.logger.info(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss = self.train_one_epoch()

            self.logger.info(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            best_score_threshold, best_final_score, summary_loss = self.validation()

            if self.do_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)


            self.logger.info( f'[RESULT]: Val. Epoch: {self.epoch}, Best Score Threshold: {best_score_threshold:.2f}, Best Score: {best_final_score:.5f}, time: {(time.time() - t):.5f}')
            if best_final_score > self.best_final_score:
                self.best_final_score = best_final_score
                self.best_score_threshold = best_score_threshold
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint.bin')
                self.save_model(f'{self.base_dir}/best-model.bin')
                self.save_predictions(f'{self.base_dir}/all_predictions.csv')

            self.early_stop(best_final_score)
            if self.early_stop_epochs > self.config.SOLVER.EARLY_STOP_PATIENCE:
                self.logger.info('Early Stopping!')
                break

            self.epoch += 1

    def validation(self):
        self.model.eval()
        t = time.time()
        summary_loss = AverageMeter()
        loss_box_reg = AverageMeter()
        loss_classifier = AverageMeter()

        self.all_predictions = []
        torch.cuda.empty_cache()
        valid_loader = tqdm(self.val_loader, total=len(self.val_loader), desc="Validating")
        with torch.no_grad():
            for step, (images, targets, image_ids) in enumerate(valid_loader):
                
                images = torch.stack(images).cuda().float()
                target_res = {}

                target_res['bbox'] = [target['boxes'].to(self.config.MODEL.DEVICE).float() for target in targets] # variable number of instances, so the entire structure can be forced to tensor
                target_res['cls'] = [target['labels'].to(self.config.MODEL.DEVICE).float() for target in targets]
                target_res['image_id'] = torch.tensor([target['image_id'] for target in targets]).to(self.config.MODEL.DEVICE).float()
                target_res['img_scale'] = torch.tensor([target['img_scale'] for target in targets]).to(self.config.MODEL.DEVICE).float()
                target_res['img_size'] = torch.tensor([(self.config.INPUT.IMG_SIZE, self.config.INPUT.IMG_SIZE) for target in targets]).to(self.config.MODEL.DEVICE).float()
  
                #print(target_res['img_scale'], target_res['img_size'])
                outputs = self.model(images, target_res)

                loss = outputs['loss']
                box_reg = outputs['box_loss']
                classifier = outputs['class_loss']
                
                summary_loss.update(scaled_loss.detach().item(), batch_size)
                loss_box_reg.update(box_reg.item(), batch_size)
                loss_classifier.update(classifier.item(), batch_size)

                inference(self.all_predictions, images, outputs['detections'], targets, image_ids)
                valid_loader.set_description(f'Validate Step {step}/{len(self.val_loader)}, ' + \
                                             f'Validation loss {summary_loss.avg:.5f}, ' + \
                                             f'time: {(time.time() - t):.5f}')
        
        best_score_threshold, best_final_score = evaluate(self.all_predictions)

        return best_score_threshold, best_final_score, summary_loss

    def train_one_epoch(self):
        self.model.train()
        summary_loss = AverageMeter()

        t = time.time()
        train_loader = tqdm(self.train_loader, total=len(self.train_loader), desc="Training")

        for step, (images, targets, image_ids, _) in enumerate(train_loader):

            images = images.to(self.device).float()
            targets = targets.to(self.device).float()
            batch_size = images.shape[0]
            
            self.optimizer.zero_grad()

            with torch.autocast(device_type=self.device):
                preds = self.model(images).squeeze()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    preds,
                    targets,
                    #TODO inserire uno scaling per lo sbilanciamento
                    #pos_weight=torch.tensor([config.POSITIVE_TARGET_WEIGHT]).to(DEVICE)
                )

            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
            self.scaler.step(self.optimizer)

            if self.scheduler is not None:
                self.scheduler.step()

            summary_loss.update(scaled_loss.detach().item(), batch_size)
            self.scaler.update()

            #Optimize con gradient accumulation
            #if ni % self.accumulate == 0:
            #    self.optimizer.step()
            #    self.optimizer.zero_grad()
                #if self.do_scheduler:
                #    self.scheduler.step(metrics=summary_loss.avg)

            
            train_loader.set_description(f'Train Step {step}/{len(self.train_loader)}, ' + \
                                         f'Learning rate {self.optimizer.param_groups[0]["lr"]}, ' + \
                                         f'summary_loss: {summary_loss.avg:.5f}, ' + \
                                         f'time: {(time.time() - t):.5f}')

        return summary_loss

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'amp_state_dict': self.amp.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_score_threshold': self.best_score_threshold,
            'best_final_score': self.best_final_score,
            'epoch': self.epoch,
        }, path)

    def save_model(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_score_threshold': self.best_score_threshold,
            'best_final_score': self.best_final_score,
        }, path)

    def save_predictions(self, path):
        df = pd.DataFrame(self.all_predictions)
        df.to_csv(path, index=False)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model, self.optimizer = self.amp.initialize(self.model, self.optimizer, opt_level='O1')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.amp = self.amp.load_state_dict(checkpoint['amp_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_score_threshold = checkpoint['best_score_threshold']
        self.best_final_score = checkpoint['best_final_score']
        self.epoch = checkpoint['epoch'] + 1

    def early_stop(self, score):
        if score < self.best_final_score:
            self.early_stop_epochs += 1
        else:
            self.early_stop_epochs = 0
