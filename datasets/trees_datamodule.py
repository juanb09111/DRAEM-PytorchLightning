
import os
import os.path
from pathlib import Path
import math
import torch
from torch.nn import functional as F
from pytorch_lightning import LightningDataModule
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random
from utils.get_anomaly_files import get_anomaly_files
from utils.id_rgb_encode_decode import id2rgb, rgb2id
from torchvision.utils import save_image
from torch.utils.data import random_split, DataLoader

import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform



torch.manual_seed(0)

class AnomalyDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transforms, split):
        
        # Read config
        self.cfg = cfg
        self.split = split   
        root = cfg.TREES_DATASET.DATASET_PATH.ROOT
        merge = True
        if split == "train":
            self.trees_used_root = os.path.join(root, split, "trees_used")
            self.imgs_root = os.path.join(root, cfg.TREES_DATASET.DATASET_PATH.RGB_TRAIN)
            if merge:
                trees_used_root_val = os.path.join(root, "val", "trees_used")
                imgs_root_val = os.path.join(root, cfg.TREES_DATASET.DATASET_PATH.RGB_VALID)
        elif split == "val":
            if merge:
                self.trees_used_root = os.path.join(root, "test", "trees_used")
                self.imgs_root = os.path.join(root, cfg.TREES_DATASET.DATASET_PATH.RGB_TEST)
            else:
                self.trees_used_root = os.path.join(root, split, "trees_used")
                self.imgs_root = os.path.join(root, cfg.TREES_DATASET.DATASET_PATH.RGB_VALID)
        elif split == "test":
            self.trees_used_root = os.path.join(root, split, "trees_used")
            self.imgs_root = os.path.join(root, cfg.TREES_DATASET.DATASET_PATH.RGB_TEST)
        
        self.imgs = get_anomaly_files(self.imgs_root)
        self.trees_used = get_anomaly_files(self.trees_used_root)
        if merge and split == "train":
            self.imgs = [*self.imgs, *get_anomaly_files(imgs_root_val)]
            self.trees_used = [*self.trees_used, *get_anomaly_files(trees_used_root_val)]

        if split != "test":
            if merge and split == "val":
                temp_split = "test"
            else:
                temp_split = split
            self.semantic_root = os.path.join(root, temp_split, cfg.TREES_DATASET.DATASET_PATH.SEMANTIC)
            self.semantic_bark_root = os.path.join(root, temp_split, cfg.TREES_DATASET.DATASET_PATH.SEMANTIC_BARK)
            self.semantic_anomaly_root = os.path.join(root, temp_split, cfg.TREES_DATASET.DATASET_PATH.SEMANTIC_ANOMALY)

            self.semantic_imgs = get_anomaly_files(self.semantic_root)
            self.semantic_imgs_bark = get_anomaly_files(self.semantic_bark_root)
            self.semantic_imgs_anomaly = get_anomaly_files(self.semantic_anomaly_root)

            if split == "train" and merge: 
                semantic_root_val = os.path.join(root, "val", cfg.TREES_DATASET.DATASET_PATH.SEMANTIC)
                semantic_bark_root_val = os.path.join(root, "val", cfg.TREES_DATASET.DATASET_PATH.SEMANTIC_BARK)
                semantic_anomaly_root_val = os.path.join(root, "val", cfg.TREES_DATASET.DATASET_PATH.SEMANTIC_ANOMALY)

                semantic_imgs_val = get_anomaly_files(semantic_root_val)
                semantic_imgs_bark_val = get_anomaly_files(semantic_bark_root_val)
                semantic_imgs_anomaly_val = get_anomaly_files(semantic_anomaly_root_val)

                self.semantic_imgs = [*self.semantic_imgs, *semantic_imgs_val]
                self.semantic_imgs_bark = [*self.semantic_imgs_bark, *semantic_imgs_bark_val]
                self.semantic_imgs_anomaly = [*self.semantic_imgs_anomaly, *semantic_imgs_anomaly_val]

        self.transforms = transforms

        if cfg.TREES_DATASET.MAX_SAMPLES != None:
            self.imgs = self.imgs[:cfg.TREES_DATASET.MAX_SAMPLES]
        

    def __getitem__(self, index):

        img_filename = self.imgs[index]

        basename = img_filename.split(".")[-2].split("/")[-1] if self.split == "test" else "_".join(img_filename.split(".")[-2].split("/")[-1].split("_")[1:])
       
        img_filename = os.path.join(img_filename)
        source_img = np.asarray(Image.open(img_filename))
        tree_used_fname = list(filter(lambda im: basename in im, self.trees_used))[0]
        tree_used = np.asarray(Image.open(tree_used_fname))
        resize = max(source_img.shape[:2]) > self.cfg.TREES_DATASET.MAX_SIZE
        if self.split != "test":
            semantic = list(filter(lambda im: basename in im, self.semantic_imgs))[0]
            semantic_bark = list(filter(lambda im: basename in im, self.semantic_imgs_bark))[0]
            semantic_anomaly = list(filter(lambda im: basename in im, self.semantic_imgs_anomaly))[0]
            
            semantic = rgb2id(np.asarray(Image.open(semantic)))
            semantic_bark = np.asarray(Image.open(semantic_bark))/255
            semantic_anomaly = np.asarray(Image.open(semantic_anomaly))/255

            if self.transforms is not None:

                apply_transforms = self.transforms(self.cfg, resize=resize)
                
                transformed = apply_transforms(
                    image=source_img,
                    masks=[semantic, semantic_bark, semantic_anomaly],
                    image0=tree_used
                )
                
            
                source_img = transformed["image"]
                semantic = transformed["masks"][-3]
                semantic_bark = transformed["masks"][-2]
                semantic_anomaly = transformed["masks"][-1]
                if len(semantic_anomaly.shape) > 2:
                    semantic_anomaly = semantic_anomaly[:,:,0]
                
                semantic_anomaly = torch.unsqueeze(semantic_anomaly, 0)
        else:
            if self.transforms is not None:

                apply_transforms = self.transforms(self.cfg, resize=resize)
                
                transformed = apply_transforms(
                    image=source_img,
                    image0=tree_used
                )

                source_img = transformed["image"]
        
        has_anomaly = "noAnomaly" not in tree_used_fname

        sample = {'image': transformed["image0"],
            "anomaly_mask": semantic_anomaly if self.split != "test" else torch.tensor(0),
            'augmented_image': transformed["image"], 
            'has_anomaly': torch.tensor(np.array([has_anomaly])), 
            'idx': torch.tensor(np.array([index])),
            "loc": tree_used_fname,
            "basename": basename
        }

        return sample
        

    def __len__(self):
        return len(self.imgs)

def get_train_transforms(cfg, resize=False):

    custom_transforms = []
    if resize:
        custom_transforms.append(A.LongestMaxSize(max_size=cfg.TREES_DATASET.MAX_SIZE))
    custom_transforms.append(A.HorizontalFlip(p=cfg.TREES_DATASET.HFLIP))
    custom_transforms.append(A.Normalize(mean=cfg.TREES_DATASET.NORMALIZE.MEAN, std=cfg.TREES_DATASET.NORMALIZE.STD))
    custom_transforms.append(ToTensorV2())

    return A.Compose(custom_transforms, additional_targets={'image0': 'image'})

def get_val_transforms(cfg, resize=False):

    custom_transforms = []
    
    if resize:
        custom_transforms.append(A.LongestMaxSize(max_size=cfg.TREES_DATASET.MAX_SIZE))
    custom_transforms.append(A.Normalize(mean=cfg.TREES_DATASET.NORMALIZE.MEAN, std=cfg.TREES_DATASET.NORMALIZE.STD))
    custom_transforms.append(ToTensorV2())

    return A.Compose(custom_transforms, additional_targets={'image0': 'image'})

def get_test_transforms(cfg, resize=False):

    custom_transforms = []
    
    if resize:
        custom_transforms.append(A.LongestMaxSize(max_size=cfg.TREES_DATASET.MAX_SIZE))
    custom_transforms.append(A.Normalize(mean=cfg.TREES_DATASET.NORMALIZE.MEAN, std=cfg.TREES_DATASET.NORMALIZE.STD))
    custom_transforms.append(ToTensorV2())

    return A.Compose(custom_transforms, additional_targets={'image0': 'image'})


def test_dataset(cfg, dataset, item, split="train"):

     # print(item)
    sample = dataset.__getitem__(item)
    image = sample["image"]
    augmented_image = sample["augmented_image"]
    # print(sample["loc"], sample["basename"])
    preview_loc = os.path.join("datasets/preview_trees/", sample["loc"])
    p = Path(preview_loc)
    p.mkdir(parents=True, exist_ok=True)
    save_image(image, os.path.join(preview_loc, sample["basename"]+".png"))
    save_image(augmented_image, os.path.join(preview_loc, sample["basename"]+"_augmented.png"))

    # Visualize Semantic
    for sem in ["anomaly_mask"]:
        print(sample[sem].shape)
        im = torch.squeeze(sample[sem])
        plt.imshow(im)
        plt.savefig(os.path.join(preview_loc, sample["basename"]+"_anomaly_mask.png"))
    

class AnomalyDataModule(LightningDataModule):
    """LightningDataModule used for training EffDet
     This supports COCO dataset input
    Args:
        cgf: config
    """

    def __init__(self, cfg, obj_name=None, test=True):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.BATCH_SIZE
        # dataset_test = AnomalyDataset(self.cfg, get_train_transforms, "val")

        # for i in range(len(dataset_test)):
        #     test_dataset(cfg, dataset_test, i, split="val")

    def train_dataset(self) -> AnomalyDataset:

        return AnomalyDataset(self.cfg, get_train_transforms, "train")

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        print("Number of training samples: {}".format(len(train_dataset)))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.cfg.TREES_DATASET.SHUFFLE,
            pin_memory=True,
            drop_last=True,
            num_workers=4,
            collate_fn=self.collate_fn_wrapper(),
        )

        return train_loader

    
    def val_dataset(self) -> AnomalyDataset:

        return AnomalyDataset(self.cfg, get_val_transforms, "val")

    def val_dataloader(self) -> DataLoader:
        val_dataset = self.val_dataset()
        print("Number of eval samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=4,
            collate_fn=self.collate_fn_wrapper(),
        )

        return val_loader


    def predict_dataset(self) -> AnomalyDataset:

        return AnomalyDataset(self.cfg, get_test_transforms, "test")

    def predict_dataloader(self) -> DataLoader:
        predict_dataset = self.predict_dataset()
        print("Number of test samples: {}".format(len(predict_dataset)))
        predict_loader = torch.utils.data.DataLoader(
            predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=4,
            collate_fn=self.collate_fn_wrapper(),
        )

        return predict_loader

    # @staticmethod
    def collate_fn_wrapper(self):
        def collate_fn(batch):
            max_width = 0
            max_height = 0
            for sample in batch:
                sample_image_shape = sample["image"].shape
                max_height = max_height if max_height > sample_image_shape[-2] else sample_image_shape[-2]
                max_width = max_width if max_width > sample_image_shape[-1] else sample_image_shape[-1]
            
            for idx, i in enumerate(batch):
                batch[idx]["original_size"] = torch.tensor([batch[idx]["image"].shape])
                padding_right = max_width - batch[idx]["image"].shape[-1]
                padding_bottom = max_height - batch[idx]["image"].shape[-2]
                p2d = (0, padding_right, 0, padding_bottom)
                batch[idx]["image"] =  F.pad(batch[idx]["image"], p2d, "constant", 0)
                batch[idx]["anomaly_mask"] =  F.pad(batch[idx]["anomaly_mask"], p2d, "constant", 0)
                if "augmented_image" in batch[idx].keys():
                    batch[idx]["augmented_image"] =  F.pad(batch[idx]["augmented_image"], p2d, "constant", 0)
            return {
                'image': torch.stack([i['image'] for i in batch]),
                'anomaly_mask': torch.stack([i['anomaly_mask'] for i in batch]),
                'augmented_image': torch.stack([i['augmented_image'] for i in batch]) if "augmented_image" in i.keys() else torch.stack([torch.tensor(0) for _ in batch ]),
                'has_anomaly': [i['has_anomaly'] for i in batch],
                'idx': torch.stack([i['idx'] for i in batch]),
                'original_size': torch.stack([i['original_size'] for i in batch ]) if 'original_size' in i.keys() else torch.stack([torch.tensor(0) for _ in batch ])
            }
        return collate_fn