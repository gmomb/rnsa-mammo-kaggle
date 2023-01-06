import os, time, warnings
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.cuda.amp import GradScaler
from sklearn.metrics import roc_auc_score

from .average import AverageMeter

from utilities.utils import optimal_f1
from modeling.optimizer import make_optimizer
from modeling.scheduler import make_scheduler
from modeling.model import LabelSmoothingCrossEntropy, SmoothBCEwLogits

warnings.filterwarnings("ignore")

class Fitter:
    def __init__(self, model, cfg, train_loader, val_loader, logger):
        self.config = cfg
        self.device = self.config.DEVICE
        self.epoch = 0
        self.train_loader = train_loader
        self.val_loader = val_loader
        #self.criterion = LabelSmoothingCrossEntropy(
        #    self.config.SOLVER.TARGET_SMOOTHING
        #)
        self.criterion = SmoothBCEwLogits(
            pos_weight=torch.Tensor([self.config.SOLVER.POS_TARGET_WEIGHT]).to(self.device),
            smoothing=self.config.SOLVER.TARGET_SMOOTHING
        )

        self.base_dir = f'{self.config.OUTPUT_DIR}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.logger = logger
        self.best_final_score = 0.0
        self.best_auc = 0.0
        self.best_score_threshold = 0.5

        self.model = model.to(self.device)

        self.optimizer = make_optimizer(self.config, self.model)
        self.scheduler = make_scheduler(self.config, self.optimizer, self.train_loader)
        self.scaler = GradScaler()

        self.logger.info(f'Fitter prepared. Device is {self.device}')
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

            t = time.time()
            best_score_threshold, best_final_score, summary_loss, best_auc = self.validation()

            if self.scheduler is not None:
                self.scheduler.step()

            self.logger.info(
                f'''[RESULT]: Val. Epoch: {self.epoch}, 
                    Best Score Threshold: {best_score_threshold:.2f}, '
                    Best Score: {best_final_score:.5f}, time: {(time.time() - t):.5f}
                    Best AUC: {best_auc:.4f}
                '''
            )
            if best_auc > self.best_auc:
                self.best_final_score = best_final_score
                self.best_score_threshold = best_score_threshold
                self.best_auc = best_auc
                self.model.eval()
                
                #Create the folder if doesnt exist and save
                Path(os.path.join(
                        self.config.OUTPUT_DIR,
                        'weights'
                )).mkdir(parents=True, exist_ok=True)

                self.save(
                    os.path.join(
                        self.config.OUTPUT_DIR,
                        'weights',
                        f'{self.config.SOLVER.MODEL_NAME}_best.pth'
                    )
                )

                #Create the folder if doesnt exist nad save
                Path(os.path.join(
                        self.config.OUTPUT_DIR,
                        'explain'
                )).mkdir(parents=True, exist_ok=True)

                self.all_predictions.to_csv(
                    os.path.join(
                        self.config.OUTPUT_DIR,
                        'explain',
                        f'fold_{self.config.INPUT.VALID_FOLD}_predictions.csv'
                    ),
                    index=False
                )

            self.epoch += 1

    def validation(self):
        self.model.eval()
        t = time.time()
        summary_loss = AverageMeter()

        fold_predictions = []
        fold_img_ids = []
        fold_targets = []
        fold_patient_ids = []

        torch.cuda.empty_cache()
        valid_loader = tqdm(self.val_loader, total=len(self.val_loader), desc="Validating")
        
        with torch.no_grad():
            for step, (images, targets, image_ids, patient_ids) in enumerate(valid_loader):
                
                images = images.to(self.device).float()
                targets = targets.to(self.device)
                batch_size = images.shape[0]

                preds = self.model(images).squeeze()

                #Add logit to the predictions
                log_preds = torch.sigmoid(preds)
                #loss = torch.nn.functional.binary_cross_entropy_with_logits(
                #    preds,
                #    targets,
                #    #TODO inserire uno scaling per lo sbilanciamento
                #    #pos_weight=torch.tensor([config.POSITIVE_TARGET_WEIGHT]).to(DEVICE)
                #)
                loss = self.criterion(preds, targets)
                
                summary_loss.update(loss.detach().item(), batch_size)

                fold_predictions.append(log_preds.detach().cpu().numpy())
                fold_img_ids.append(image_ids.detach().cpu().numpy())
                fold_targets.append(targets.detach().cpu().numpy())
                fold_patient_ids.append(patient_ids.detach().cpu().numpy())

                #TODO: creare un layer di inferenza
                #inference(self.all_predictions, images, outputs['detections'], targets, image_ids)
                valid_loader.set_description(
                    f'Validate Step {step}/{len(self.val_loader)}, ' + \
                    f'Validation loss {summary_loss.avg:.2f}, ' + \
                    f'time: {(time.time() - t):.2f}' + \
                    f'Best score found: {self.best_final_score:.4f}' + \
                    f'Best AUC found: {self.best_auc:.4f}'
                )
        
        fold_predictions = np.concatenate(fold_predictions)
        fold_img_ids = np.concatenate(fold_img_ids)
        fold_targets = np.concatenate(fold_targets)
        fold_patient_ids = np.concatenate(fold_patient_ids)

        best_final_score, best_score_threshold = optimal_f1(fold_targets, fold_predictions)

        self.all_predictions = pd.DataFrame({
            'patient_ids': fold_patient_ids,
            'img_ids': fold_img_ids,
            'label': fold_targets,
            'preds': fold_predictions,
            'best_threshold': best_score_threshold,
        })

        auc_score = roc_auc_score(self.all_predictions['label'], self.all_predictions['preds'])
        return best_score_threshold, best_final_score, summary_loss, auc_score

    def train_one_epoch(self):
        self.model.train()
        summary_loss = AverageMeter()

        t = time.time()
        train_loader = tqdm(self.train_loader, total=len(self.train_loader), desc="Training")

        for step, (images, targets, _, _) in enumerate(train_loader):

            images = images.to(self.device).float()
            targets = targets.to(self.device)
            batch_size = images.shape[0]
            
            self.optimizer.zero_grad()

            with torch.autocast(device_type=self.device):
                preds = self.model(images).squeeze()
                loss = self.criterion(preds, targets)
                #loss = torch.nn.functional.binary_cross_entropy_with_logits(
                #    preds,
                #    targets,
                #    #TODO inserire uno scaling per lo sbilanciamento
                #    pos_weight=torch.tensor([self.config.SOLVER.POS_TARGET_WEIGHT]).to(self.device)
                #)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            summary_loss.update(loss.detach().item(), batch_size)
            self.scaler.update()

            #Optimize con gradient accumulation
            #if ni % self.accumulate == 0:
            #    self.optimizer.step()
            #    self.optimizer.zero_grad()
                #if self.do_scheduler:
                #    self.scheduler.step(metrics=summary_loss.avg)

            
            train_loader.set_description(
                f'Train Step {step}/{len(self.train_loader)}, ' + \
                f'Learning rate {self.optimizer.param_groups[0]["lr"]}, ' + \
                f'summary_loss: {summary_loss.avg:.5f}, ' + \
                f'time: {(time.time() - t):.5f}'
            )

        return summary_loss

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_score_threshold': self.best_score_threshold,
            'best_final_score': self.best_final_score,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_score_threshold = checkpoint['best_score_threshold']
        self.best_final_score = checkpoint['best_final_score']
        self.epoch = checkpoint['epoch'] + 1