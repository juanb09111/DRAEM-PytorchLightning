import torch
import numpy as np
import os
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor
)
from pytorch_lightning.utilities.model_summary  import ModelSummary
from pytorch_lightning import loggers as pl_loggers
from detectron2.config import get_cfg
from detectron2.utils.events import _CURRENT_STORAGE_STACK, EventStorage


# from pandepth import DeepLabV3, FCN
from utils.add_custom_params import add_custom_params

from datasets.anomaly_datamodule import AnomalyDataModule
# from datasets.anomaly_datamodule4 import AnomalyDataModule
# from datasets.anomaly_data_module_real import AnomalyDataModule



def train(args):
    
    # Retrieve Config and and custom base parameter
    cfg = get_cfg()
    add_custom_params(cfg)
    cfg.merge_from_file(args.config)
    cfg.NUM_GPUS = torch.cuda.device_count()
    
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    logger = logging.getLogger("pytorch_lightning.core")
    _CURRENT_STORAGE_STACK.append(EventStorage())

    for obj_name in args.obj_names:
        datamodule = AnomalyDataModule(cfg, obj_name)
   

        checkpoint_path = cfg.CHECKPOINT_PATH_INFERENCE if (args.predict or args.eval) else cfg.CHECKPOINT_PATH_TRAINING
        
        # if args.model_name == "deeplab":
        #     model = DeepLabV3
        # elif args.model_name == "fcn":
        #     model = FCN
        # # Create model or load a checkpoint
        # if os.path.exists(checkpoint_path):
        #     print('""""""""""""""""""""""""""""""""""""""""""""""')
        #     print("Loading model from {}".format(checkpoint_path))
        #     print('""""""""""""""""""""""""""""""""""""""""""""""')
        #     semantic = model.load_from_checkpoint(cfg=cfg,
        #         checkpoint_path=checkpoint_path)
        # else:
        #     print('""""""""""""""""""""""""""""""""""""""""""""""')
        #     print("Creating a new model")
        #     print('""""""""""""""""""""""""""""""""""""""""""""""')
        #     semantic = model(cfg)
        #     cfg.CHECKPOINT_PATH_TRAINING = None
        #     cfg.CHECKPOINT_PATH_INFERENCE = None

        # # logger.info(efficientps.print)
        # ModelSummary(semantic, max_depth=-1)
        # # Callbacks / Hooks
        # early_stopping = EarlyStopping('IoU', patience=10, mode='max')
        # checkpoint = ModelCheckpoint(monitor='IoU',
        #                             mode='max',
        #                             dirpath=cfg.CALLBACKS.CHECKPOINT_DIR,
        #                             save_last=True,
        #                             verbose=True)

        # lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # exp_name = "{}_train".format(args.exp_name) if args.predict else args.exp_name
        # tb_logger = pl_loggers.TensorBoardLogger("tb_logs", name=exp_name)
        # # Create a pytorch lighting trainer
        # trainer = pl.Trainer(
        #     # weights_summary='full',
        #     logger=tb_logger,
        #     auto_lr_find=args.tune,
        #     log_every_n_steps=np.floor(len(datamodule.val_dataloader())/(cfg.BATCH_SIZE*torch.cuda.device_count())) -1,
        #     devices=1 if args.tune else list(range(torch.cuda.device_count())),
        #     strategy=None if args.tune else "ddp",
        #     accelerator='gpu',
        #     num_sanity_val_steps=0,
        #     fast_dev_run=cfg.SOLVER.FAST_DEV_RUN if args.fast_dev else False,
        #     callbacks=[early_stopping, checkpoint, lr_monitor],
        #     # precision=cfg.PRECISION,
        #     resume_from_checkpoint=cfg.CHECKPOINT_PATH_INFERENCE if (args.predict or args.eval) else cfg.CHECKPOINT_PATH_TRAINING,
        #     # gradient_clip_val=0,
        #     accumulate_grad_batches=cfg.SOLVER.ACCUMULATE_GRAD
        # )
        # logger.addHandler(logging.StreamHandler())
        # if args.tune:
        #     lr_finder = trainer.tuner.lr_find(semantic, datamodule, min_lr=1e-4, max_lr=0.1, num_training=100)
        #     print("LR found:", lr_finder.suggestion())
        # elif args.predict:
        #     semantic.eval()
        #     with torch.no_grad():
        #         trainer.predict(semantic, datamodule)
        # elif args.eval:
        #     semantic.eval()
        #     with torch.no_grad():
        #         trainer.validate(semantic, datamodule)
        # else:
        #     trainer.fit(semantic, datamodule)