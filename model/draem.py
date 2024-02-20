import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import pil_to_tensor
import pytorch_lightning as pl
from .sub_networks import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from .loss import FocalLoss, SSIM
import numpy as np
from skimage import measure
from torchmetrics import JaccardIndex
# from kornia.color import lab_to_rgb
from PIL import Image
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

        self.valid_acc = JaccardIndex(num_classes=2, ignore_index=0)
        self.rates = {
            "FP": 0,
            "FN": 0,
            "TP": 0,
            "TN": 0
        }

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
        
    def keep_n_largest_connected_components(self, binary_mask, n, min_size):
        # Convert the PyTorch tensor to a NumPy array
        binary_mask_np = binary_mask.cpu().numpy()

        # Find connected components using scikit-image's label function
        labeled_mask, num_components = measure.label(binary_mask_np, connectivity=2, return_num=True)

        # Identify the n largest connected components
        largest_component_labels = self.find_n_largest_connected_components(labeled_mask, num_components, n, min_size)

        # Zero out all other regions except the n largest connected components
        result_mask = torch.tensor(np.isin(labeled_mask, largest_component_labels), dtype=torch.float32)

        return result_mask

    def find_n_largest_connected_components(self, labeled_mask, num_components, n, min_size):
        # Calculate the size of each connected component
        component_sizes = measure.regionprops(labeled_mask)

        # Filter components based on the minimum size
        valid_components = [prop for prop in component_sizes if prop.area >= min_size]

        # Sort the connected components by size in descending order
        sorted_components = sorted(valid_components, key=lambda prop: prop.area, reverse=True)

        # Get the labels of the n largest connected components
        largest_component_labels = [prop.label for prop in sorted_components[:n]]

        return largest_component_labels
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        
        while batch_idx <= 16000:
            gray_batch = batch["image"]
            gray_rec = self.reconstructive(gray_batch)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)
            

            out_mask = self.discriminative(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            t_mask = batch["anomaly_mask"]
            # print(gray_batch.shape, t_mask.shape)

            for idx, _ in enumerate(batch["image"]): 
                tb_logger = self.logger.experiment
                
                im = batch["image"][idx]
                im[0,:,:] =  batch["image"][idx][2,:,:] 
                im[2,:,:] =  batch["image"][idx][0,:,:] 
                mask = out_mask_sm[idx]
                gt_mask = t_mask[idx]
                g_rec = gray_rec[idx]
                g_rec[0,:,:] =  gray_rec[idx][2,:,:] 
                g_rec[2,:,:] =  gray_rec[idx][0,:,:]
                t_ch = torch.unsqueeze(torch.zeros_like(mask[0]), 0)

                mask[1] = torch.where(mask[1] > 0.5, 1, 0)
                mask[1] = self.keep_n_largest_connected_components(mask[1], 3, 40)

                # Calculate TP, FP, TN, FN

                gt_vals = torch.unique(gt_mask)
                gt_has_anomaly = 1 in gt_vals
                mask_vals = torch.unique(mask[1])
                pred_has_anomaly = 1 in mask_vals
                # print(torch.unsqueeze(mask[1], 0).shape, gt_mask.long().shape)
                jaccard_score = self.valid_acc(torch.unsqueeze(mask[1], 0), gt_mask.long())
                
                # if pred_has_anomaly and gt_has_anomaly and jaccard_score > 0.2:
                if pred_has_anomaly and gt_has_anomaly:
                    self.rates["TP"] +=1
                elif not pred_has_anomaly and gt_has_anomaly:
                    self.rates["FN"] +=1
                elif pred_has_anomaly and not gt_has_anomaly:
                    self.rates["FP"] +=1
                elif not pred_has_anomaly and not gt_has_anomaly:
                    self.rates["TN"] +=1
                # -------------------------------------------
                heatmap = torch.cat((torch.unsqueeze(mask[1], 0), t_ch, torch.unsqueeze(mask[0], 0)))

                # anomaly_mask = torch.where(mask[1] > 0.5, 1, 0)
                # img = TF.to_pil_image(lab_to_rgb(im))  
                img = TF.to_pil_image(im)
                h_img = TF.to_pil_image(heatmap)

                res = Image.blend(img, h_img, 0.5)
                
                # tb_logger.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/image_".format(dataloader_idx, batch_idx, idx), lab_to_rgb(im))
                # tb_logger.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/g_rec_".format(dataloader_idx, batch_idx, idx), lab_to_rgb(g_rec))
                tb_logger.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/image_".format(dataloader_idx, batch_idx, idx), im)
                tb_logger.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/g_rec_".format(dataloader_idx, batch_idx, idx), g_rec)
                tb_logger.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/out_mask_sm_iou_{}".format(dataloader_idx, batch_idx, idx, jaccard_score), pil_to_tensor(res))
                tb_logger.add_image("dl_idx_{}_batch_idx_{}_sample_idx_{}/t_mask".format(dataloader_idx, batch_idx, idx), gt_mask)
            break
    
    def on_predict_epoch_end(self, results):
        
        if self.rates["TP"] == 0:
            print("TP: ", self.rates["TP"])
        else:
            print("TP: ", self.rates["TP"], self.rates["TP"]/(self.rates["TP"] + self.rates["FN"]))

        
        if self.rates["FP"] == 0:
            print("FP: ", self.rates["FP"])
        else:
            print("FP: ", self.rates["FP"], self.rates["FP"]/(self.rates["FP"] + self.rates["TN"]))

        if self.rates["TN"] == 0:
            print("TN: ", self.rates["TN"])
        else:
            print("TN: ", self.rates["TN"], self.rates["TN"]/(self.rates["TN"] + self.rates["FP"]))
        

        if self.rates["FN"] == 0:
            print("FN: ", self.rates["FN"])
        else:
            print("FN: ", self.rates["FN"], self.rates["FN"]/(self.rates["FN"] + self.rates["TP"]))
        

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
