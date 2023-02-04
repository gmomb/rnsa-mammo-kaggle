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
    def __init__(self, model, cfg, train_loader, val_loader, logger, neptune_runner):
        self.config = cfg
        self.device = self.config.DEVICE
        
        self.epoch = 0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.neptune_runner = neptune_runner
        self.criterion = SmoothBCEwLogits(
            pos_weight=torch.Tensor([self.config.SOLVER.POS_TARGET_WEIGHT]).to(self.device),
            smoothing=self.config.SOLVER.TARGET_SMOOTHING
        )
        
        self.aux_criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=self.config.SOLVER.TARGET_SMOOTHING
        )

        self.base_dir = f'{self.config.OUTPUT_DIR}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.logger = logger
        self.best_final_score = 0.0
        self.best_score_threshold = 0.5
        self.best_auc = 0.0

        self.model = model.to(self.config.DEVICE)

        self.optimizer = make_optimizer(self.config, self.model)
        self.scheduler = make_scheduler(self.config, self.optimizer, self.train_loader)
        self.scaler = GradScaler()

        self.logger.info(f'Fitter prepared. Device is {self.device}')
        self.do_scheduler = False
        self.logger.info("Start training")

    def fit(self):
        for epoch in range(self.epoch, self.config.SOLVER.MAX_EPOCHS ):
            
            #TODO: inserire warmup
            t = time.time()
            summary_loss, cancer_loss, aux_loss = self.train_one_epoch()

            self.logger.info(
                f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, cancer_loss: {cancer_loss.avg:.5f}, aux_loss: {aux_loss.avg:.5f}, time: {(time.time() - t):.5f}'
            )

            t = time.time()
            best_score_threshold, best_final_score, summary_loss, valid_auc = self.validation()

            if self.scheduler is not None:
                self.scheduler.step()

            self.logger.info( f'[RESULT]: Val. Epoch: {self.epoch}, Best Score Threshold: {best_score_threshold:.2f}, Best Score: {best_final_score:.5f}, time: {(time.time() - t):.5f}')
            if valid_auc > self.best_auc:
                print(f'Best score found: fbeta {best_final_score:.4f}')
                self.best_final_score = best_final_score
                self.best_score_threshold = best_score_threshold
                self.best_auc = valid_auc
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
                        f'{self.config.SOLVER.MODEL_NAME}_f{self.config.INPUT.VALID_FOLD}.pth'
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
        cancer_loss_meter = AverageMeter()
        aux_loss_meter = AverageMeter()

        fold_predictions = []
        fold_img_ids = []
        fold_targets = []
        fold_patient_ids = []
        fold_site_ids = []
        fold_laterality = []

        torch.cuda.empty_cache()
        valid_loader = tqdm(self.val_loader, total=len(self.val_loader), desc="Validating")
        
        with torch.no_grad():
            for step, (images, targets, aux_targets, imgs_meta) in enumerate(valid_loader):
                
                images = images.to(self.device).float()
                targets = targets.to(self.device)
                aux_targets = aux_targets.to(self.device)
                batch_size = images.shape[0]
                
                #il modello ritorna pred e aux_pred
                preds, aux_preds = self.model(images)

                #Add logit to the predictions
                log_preds = torch.sigmoid(preds).squeeze()
                #loss = torch.nn.functional.binary_cross_entropy_with_logits(
                #    preds,
                #    targets,
                #    #TODO inserire uno scaling per lo sbilanciamento
                #    #pos_weight=torch.tensor([config.POSITIVE_TARGET_WEIGHT]).to(DEVICE)
                #)
                loss = self.criterion(preds.squeeze(), targets)
                
                aux_losses = []
                for cnt, pred in enumerate(self.config.INPUT.AUX_TARGETS):
                    aux_losses.append(
                        self.aux_criterion(aux_preds[cnt], aux_targets[:, cnt])
                    )
                    
                aux_loss = torch.stack(aux_losses).mean()
                final_loss = loss + self.config.SOLVER.AUX_FACTOR * aux_loss
                
                summary_loss.update(final_loss.detach().item(), batch_size)
                cancer_loss_meter.update(loss.detach().item(), batch_size)
                aux_loss_meter.update(aux_loss.detach().item(), batch_size)
                
                fold_predictions.append(log_preds.detach().cpu().numpy())
                fold_targets.append(targets.detach().cpu().numpy())
                
                fold_patient_ids.append(imgs_meta['patient_id'].detach().cpu().numpy())
                fold_img_ids.append(imgs_meta['img_id'].detach().cpu().numpy())
                fold_site_ids.append(imgs_meta['site_id'].detach().cpu().numpy())
                fold_laterality.append(np.array(imgs_meta['laterality']))

                #TODO: creare un layer di inferenza
                #inference(self.all_predictions, images, outputs['detections'], targets, image_ids)

                self.neptune_runner["valid/batch/total_loss"].log(final_loss.detach().item())
                self.neptune_runner["valid/batch/cancer_loss"].log(loss.detach().item())
                self.neptune_runner["valid/batch/aux_loss"].log(aux_loss.detach().item())

                valid_loader.set_description(
                    f'Validate Step {step}/{len(self.val_loader)}, ' + \
                    f'Validation loss {summary_loss.avg:.5f}, ' + \
                    f'Validation cancer_loss {cancer_loss_meter.avg:.5f}, ' + \
                    f'Validation aux_loss {aux_loss_meter.avg:.5f}, ' + \
                    f'time: {(time.time() - t):.5f}' + \
                    f'Best score found: {self.best_final_score:.4f}'
                )
        
        fold_predictions = np.concatenate(fold_predictions)
        fold_targets = np.concatenate(fold_targets)
        
        fold_img_ids = np.concatenate(fold_img_ids)
        fold_patient_ids = np.concatenate(fold_patient_ids)
        fold_site_ids = np.concatenate(fold_site_ids)
        fold_laterality = np.concatenate(fold_laterality)

        self.all_predictions = pd.DataFrame({
            'laterality': fold_laterality,
            'site_ids': fold_site_ids,
            'patient_ids': fold_patient_ids,
            'img_ids': fold_img_ids,
            'label': fold_targets,
            'preds': fold_predictions,
        })
        
        
        best_final_score, best_score_threshold = optimal_f1(
            self.all_predictions['label'], self.all_predictions['preds']
        )
        
        auc = roc_auc_score(self.all_predictions['label'], self.all_predictions['preds'])
        print(f'Best score found at AUC {auc}: {best_final_score} with threshold {best_score_threshold}')

        self.neptune_runner["valid/auc_score"].log(auc)
        self.neptune_runner["valid/betaf1_best"].log(best_final_score)
        self.neptune_runner["valid/betaf1_thr"].log(best_score_threshold)

        return best_score_threshold, best_final_score, summary_loss, auc

    def train_one_epoch(self):
        self.model.train()
        summary_loss = AverageMeter()
        cancer_loss_meter = AverageMeter()
        aux_loss_meter = AverageMeter()

        t = time.time()
        train_loader = tqdm(self.train_loader, total=len(self.train_loader), desc="Training")

        for step, (images, targets, aux_targets, _) in enumerate(train_loader):

            images = images.to(self.device).float()
            targets = targets.to(self.device)
            aux_targets = aux_targets.to(self.device)
            batch_size = images.shape[0]
            
            self.optimizer.zero_grad()
            
            aux_losses = []
            with torch.autocast(device_type=self.device):
                preds, aux_preds = self.model(images)
                loss = self.criterion(preds.squeeze(), targets)
                
                for cnt, pred in enumerate(self.config.INPUT.AUX_TARGETS):
                    aux_losses.append(
                        self.aux_criterion(aux_preds[cnt], aux_targets[:, cnt])
                    )
                    
                aux_loss = torch.stack(aux_losses).mean()
            
            final_loss = loss + self.config.SOLVER.AUX_FACTOR * aux_loss
            
            self.scaler.scale(final_loss).backward()
            self.scaler.step(self.optimizer)

            self.neptune_runner["train/batch/total_loss"].log(final_loss.detach().item())
            self.neptune_runner["train/batch/cancer_loss"].log(loss.detach().item())
            self.neptune_runner["train/batch/aux_loss"].log(aux_loss.detach().item())

            summary_loss.update(final_loss.detach().item(), batch_size)
            cancer_loss_meter.update(loss.detach().item(), batch_size)
            aux_loss_meter.update(aux_loss.detach().item(), batch_size)
            
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
                f'cancer_loss: {cancer_loss_meter.avg:.5f}, ' + \
                f'aux_loss: {aux_loss_meter.avg:.5f}, ' + \
                f'time: {(time.time() - t):.5f}'
            )

        return summary_loss, cancer_loss_meter, aux_loss_meter

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