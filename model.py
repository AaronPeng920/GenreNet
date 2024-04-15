import lightning.pytorch as pl
import torch
import torch.nn as nn
import os
from ema import EMA
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
import numpy as np

class RoleNet(pl.LightningModule):
    def __init__(self, model, **params):
        super().__init__()
        
        self.model = model
        
        self.params = params
        self.criterion = nn.CrossEntropyLoss()
        
        # If you want to save hyperparameters then anti-comment it
        #  self.save_hyperparameters()
            
    def training_step(self, batch, batch_idx):
        waveforms, spectrograms, labels = batch
        spectrograms = spectrograms.squeeze(1)
        preds = self.model(waveforms, spectrograms)
        train_loss = self.criterion(preds, labels)
        self.log_dict({"train_loss": train_loss.data, "lr": self.optimizer.state_dict()['param_groups'][0]['lr']}, sync_dist=True)
        return train_loss
    
    def on_train_start(self):
        self.model_ema = EMA(self.model, self.params['ema_decay'])
        self.model_ema.register()
    
    def on_train_batch_end(self, *args, **kwargs):
        self.model_ema.update()
    
    def on_validation_epoch_start(self):
        self.model_ema.apply_shadow()
        self.val_epoch_batchses = 0
        self.val_epoch_accuracy_songs = 0
    
    def on_validation_epoch_end(self):
        self.model_ema.restore()
        accuracy_rate = self.val_epoch_accuracy_songs / self.val_epoch_batchses
        self.log("accuracy", accuracy_rate, sync_dist=True)
        
    def validation_step(self, batch, batch_idx):
        waveforms, spectrograms, labels = batch
        spectrograms = spectrograms.squeeze(1)
        preds = self.model(waveforms, spectrograms)
        val_loss = self.criterion(preds, labels)
        pred_label = torch.max(preds, dim=-1)[1]
        
        vote = np.zeros(self.params['num_classes'])
        for i in pred_label:
            vote[i] += 1
        answer = np.argmax(vote, 0)
        self.val_epoch_accuracy_songs += 1 if answer == labels[0] else 0
        self.val_epoch_batchses += 1
        
        self.log_dict({"val_loss": val_loss.data}, sync_dist=True)
        
    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.params['lr'], momentum=0.9, weight_decay=1e-04, nesterov=True
            )
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.params['milestones'], gamma=self.params['gamma'])
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

class VGG(pl.LightningModule):
    def __init__(self, model, **params):
        super().__init__()
        
        self.model = model
        
        self.params = params
        self.criterion = nn.CrossEntropyLoss()
        
        # If you want to save hyperparameters then anti-comment it
        #  self.save_hyperparameters()
            
    def training_step(self, batch, batch_idx):
        waveforms, spectrograms, labels = batch
        spectrograms = spectrograms.unsqueeze(1)
        preds = self.model(spectrograms)
        train_loss = self.criterion(preds, labels)
        self.log_dict({"train_loss": train_loss.data, "lr": self.optimizer.state_dict()['param_groups'][0]['lr']}, sync_dist=True)
        return train_loss
    
    def on_validation_epoch_start(self):
        self.val_epoch_batchses = 0
        self.val_epoch_accuracy_songs = 0
    
    def on_validation_epoch_end(self):
        accuracy_rate = self.val_epoch_accuracy_songs / self.val_epoch_batchses
        self.log("accuracy", accuracy_rate, sync_dist=True)
        
    def validation_step(self, batch, batch_idx):
        waveforms, spectrograms, labels = batch
        spectrograms = spectrograms.unsqueeze(1)
        preds = self.model(spectrograms)
        val_loss = self.criterion(preds, labels)
        pred_label = torch.max(preds, dim=-1)[1]
        
        vote = np.zeros(self.params['num_classes'])
        for i in pred_label:
            vote[i] += 1
        answer = np.argmax(vote, 0)
        self.val_epoch_accuracy_songs += 1 if answer == labels[0] else 0
        self.val_epoch_batchses += 1
        
        self.log_dict({"val_loss": val_loss.data}, sync_dist=True)
        
    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params['lr'], momentum=self.params['momentum'], nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.params['milestones'], gamma=self.params['gamma'])
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler, "monitor": "accuracy"}
    
class ResNet(pl.LightningModule):
    def __init__(self, model, **params):
        super().__init__()
        
        self.model = model
        
        self.params = params
        self.criterion = nn.CrossEntropyLoss()
        
        # If you want to save hyperparameters then anti-comment it
        #  self.save_hyperparameters()
            
    def training_step(self, batch, batch_idx):
        waveforms, spectrograms, labels = batch
        spectrograms = spectrograms.unsqueeze(1)
        preds = self.model(spectrograms)
        train_loss = self.criterion(preds, labels)
        self.log_dict({"train_loss": train_loss.data, "lr": self.optimizer.state_dict()['param_groups'][0]['lr']}, sync_dist=True)
        return train_loss
    
    def on_validation_epoch_start(self):
        self.val_epoch_batchses = 0
        self.val_epoch_accuracy_songs = 0
    
    def on_validation_epoch_end(self):
        accuracy_rate = self.val_epoch_accuracy_songs / self.val_epoch_batchses
        self.log("accuracy", accuracy_rate, sync_dist=True)
        
    def validation_step(self, batch, batch_idx):
        waveforms, spectrograms, labels = batch
        spectrograms = spectrograms.unsqueeze(1)
        preds = self.model(spectrograms)
        val_loss = self.criterion(preds, labels)
        pred_label = torch.max(preds, dim=-1)[1]
        
        vote = np.zeros(self.params['num_classes'])
        for i in pred_label:
            vote[i] += 1
        answer = np.argmax(vote, 0)
        self.val_epoch_accuracy_songs += 1 if answer == labels[0] else 0
        self.val_epoch_batchses += 1
        
        self.log_dict({"val_loss": val_loss.data}, sync_dist=True)
        
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler, "monitor": "accuracy"}
    
class AlexNet(pl.LightningModule):
    def __init__(self, model, **params):
        super().__init__()
        
        self.model = model
        
        self.params = params
        self.criterion = nn.CrossEntropyLoss()
        
        # If you want to save hyperparameters then anti-comment it
        #  self.save_hyperparameters()
            
    def training_step(self, batch, batch_idx):
        waveforms, spectrograms, labels = batch
        spectrograms = spectrograms.unsqueeze(1)
        preds = self.model(spectrograms)
        train_loss = self.criterion(preds, labels)
        self.log_dict({"train_loss": train_loss.data, "lr": self.optimizer.state_dict()['param_groups'][0]['lr']}, sync_dist=True)
        return train_loss
    
    def on_validation_epoch_start(self):
        self.val_epoch_batchses = 0
        self.val_epoch_accuracy_songs = 0
    
    def on_validation_epoch_end(self):
        accuracy_rate = self.val_epoch_accuracy_songs / self.val_epoch_batchses
        self.log("accuracy", accuracy_rate, sync_dist=True)
        
    def validation_step(self, batch, batch_idx):
        waveforms, spectrograms, labels = batch
        spectrograms = spectrograms.unsqueeze(1)
        preds = self.model(spectrograms)
        val_loss = self.criterion(preds, labels)
        pred_label = torch.max(preds, dim=-1)[1]
        
        vote = np.zeros(self.params['num_classes'])
        for i in pred_label:
            vote[i] += 1
        answer = np.argmax(vote, 0)
        self.val_epoch_accuracy_songs += 1 if answer == labels[0] else 0
        self.val_epoch_batchses += 1
        
        self.log_dict({"val_loss": val_loss.data}, sync_dist=True)
        
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.params['lr'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.params['step'], gamma=self.params['gamma'])
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler, "monitor": "accuracy"}
       

