import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from .sub_networks import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from .loss import FocalLoss, SSIM

import os.path



class DRAEM(pl.LightningModule):
   
    def __init__(self, cfg):
        super(DRAEM, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = cfg.SOLVER.BASE_LR
        self.cfg = cfg


        self.reconstructive = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        self.discriminative = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        
        #Loss functions
        self.l2 = nn.modules.loss.MSELoss()
        self.ssim_loss = SSIM()
        self.focal_loss = FocalLoss()

        self.automatic_optimization = False


    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        predictions = self.shared_step(x)
        return predictions

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        
        opt = self.optimizers()
        
        preds = self.shared_step(batch)
        #loss
        gray_batch = batch["image"]
        anomaly_mask = batch["anomaly_mask"]

        l2_loss = self.l2(preds["gray_rec"],gray_batch)
        ssim_loss = self.ssim_loss(preds["gray_rec"], gray_batch)

        segment_loss = self.focal_loss(preds["out_mask_sm"], anomaly_mask)
        loss_sum = l2_loss + ssim_loss + segment_loss

        # Optimize
        opt.zero_grad()
        self.manual_backward(loss_sum)
        opt.step()

        loss_dict = dict()
        loss_dict.update({"l2_loss":l2_loss , "ssim_loss":ssim_loss , "segment_loss":segment_loss , "loss_sum":loss_sum })
        # Add losses to logs
        [self.log(k, v, batch_size=self.cfg.BATCH_SIZE, on_step=False, on_epoch=True, sync_dist=True) for k,v in loss_dict.items()]
        self.log('train_loss', loss_sum, batch_size=self.cfg.BATCH_SIZE, on_step=True, on_epoch=False, sync_dist=False)
        self.log('train_loss_epoch', loss_sum, batch_size=self.cfg.BATCH_SIZE, on_step=False, on_epoch=True, sync_dist=True)
        return {'loss': loss_sum}

    def shared_step(self, inputs):

        loss = dict()
        predictions = dict()
        aug_gray_batch = inputs["augmented_image"]

        gray_rec = self.reconstructive(aug_gray_batch)
        joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

        out_mask = self.discriminative(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)

        predictions.update({"out_mask_sm": out_mask_sm, "gray_rec": gray_rec})
        return predictions
        

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        
        predictions = self.shared_step(batch)
        return predictions
        
    
    def on_predict_epoch_end(self, results):
       
       return results
        

    def configure_optimizers(self):
        print("Optimizer - using {} with lr {}".format(self.cfg.SOLVER.NAME, self.cfg.SOLVER.BASE_LR))

        optimizer = torch.optim.Adam([
                                      {"params": self.reconstructive.parameters(), "lr": self.learning_rate},
                                      {"params": self.discriminative.parameters(), "lr": self.learning_rate}
                                      ])

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[self.cfg.SOLVER.N_EPOCHS*0.8,self.cfg.SOLVER.N_EPOCHS*0.9],gamma=0.2, last_epoch=-1)


        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
