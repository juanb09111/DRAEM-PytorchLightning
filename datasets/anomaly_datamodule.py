import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, cfg, root_dir, transforms, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.png"))
        self.resize_shape=resize_shape
        self.transforms = transforms
        self.cfg = cfg

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        # image = np.transpose(image, (2, 0, 1))
        # mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)


        apply_transforms = self.transforms(self.cfg)
        
        transformed = apply_transforms(
            image=image,
            masks=[mask]
        )

        basename = self.images[idx].split("/")[-1].split(".")[0]
        loc = "/".join(self.images[idx].split("/")[1:-1])

        sample = {'image': transformed["image"],
            'has_anomaly': torch.tensor(np.array([has_anomaly])),
            'mask': transformed["masks"][0], 
            'idx': torch.tensor(np.array([idx])),
            'basename': basename,
            "loc": loc
        }

        return sample



class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, cfg, root_dir, anomaly_source_path, transforms, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))

        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

        self.transforms = transforms
        self.cfg = cfg


    def __len__(self):
        return len(self.image_paths)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        # augmented_image = np.transpose(augmented_image, (2, 0, 1))
        # image = np.transpose(image, (2, 0, 1))
        # anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx], self.anomaly_source_paths[anomaly_source_idx])
        
        basename = self.image_paths[idx].split("/")[-1].split(".")[0]
        loc = "/".join(self.image_paths[idx].split("/")[1:-1])

        
        apply_transforms = self.transforms(self.cfg)

        transformed = apply_transforms(
            image=image,
            masks=[anomaly_mask]
        )

        transformed_augmented = apply_transforms(image=augmented_image)  

        sample = {'image': transformed["image"],
            "anomaly_mask": transformed["masks"][-1],
            'augmented_image': transformed_augmented["image"], 
            'has_anomaly': torch.tensor(np.array([has_anomaly])), 
            'idx': torch.tensor(np.array([idx])),
            "loc": loc,
            "basename": basename
        }

        return sample



def get_train_transforms(cfg):

    custom_transforms = []
    custom_transforms.append(ToTensorV2())

    return A.Compose(custom_transforms)


def get_test_transforms(cfg):

    custom_transforms = []
    custom_transforms.append(ToTensorV2())

    return A.Compose(custom_transforms)


def test_dataset(cfg, dataset, item):

    # print(item)
    sample = dataset.__getitem__(item)
    image = sample["image"]
    augmented_image = sample["augmented_image"]
    # print(sample["loc"], sample["basename"])
    preview_loc = os.path.join("datasets/preview/", sample["loc"])
    p = Path(preview_loc)
    p.mkdir(parents=True, exist_ok=True)
    save_image(image, os.path.join(preview_loc, sample["basename"]+".png"))
    save_image(augmented_image, os.path.join(preview_loc, sample["basename"]+"_augmented.png"))

    #Visualize Semantic
    for sem in ["anomaly_mask"]:
        im = sample[sem]
        plt.imshow(im)
        plt.savefig(os.path.join(preview_loc, sample["basename"]+"_anomaly_mask.png"))
    


class AnomalyDataModule(LightningDataModule):
    """LightningDataModule used for training EffDet
     This supports COCO dataset input
    Args:
        cgf: config
    """

    def __init__(self, cfg, obj_name, test=True):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.BATCH_SIZE
        self.obj_name = obj_name

        if test:
            root_dir = os.path.join(self.cfg.ANOMALY_DATASET.DATASET_PATH.ROOT, self.cfg.ANOMALY_DATASET.DATASET_PATH.MVTEC, self.obj_name, "train/good/")
            anomaly_source_path = os.path.join(self.cfg.ANOMALY_DATASET.DATASET_PATH.ROOT, 
                self.cfg.ANOMALY_DATASET.DATASET_PATH.ANOMALY_SOURCE)
            
            resize_shape = (self.cfg.ANOMALY_DATASET.RESIZE.HEIGHT, self.cfg.ANOMALY_DATASET.RESIZE.WIDTH)
            dataset_test = MVTecDRAEMTrainDataset(self.cfg, root_dir, anomaly_source_path, get_train_transforms, resize_shape)

            for i in range(10):
                test_dataset(cfg, dataset_test, i)


    def train_dataset(self) -> MVTecDRAEMTrainDataset:
        root_dir = os.path.join(self.cfg.ANOMALY_DATASET.DATASET_PATH.ROOT, self.cfg.ANOMALY_DATASET.DATASET_PATH.MVTEC, self.obj_name, "train/good/")
        anomaly_source_path = os.path.join(self.cfg.ANOMALY_DATASET.DATASET_PATH.ROOT, 
            self.cfg.ANOMALY_DATASET.DATASET_PATH.ANOMALY_SOURCE)
        
        resize_shape = (self.cfg.ANOMALY_DATASET.RESIZE.HEIGHT, self.cfg.ANOMALY_DATASET.RESIZE.WIDTH)
        return MVTecDRAEMTrainDataset(self.cfg, root_dir, anomaly_source_path, get_train_transforms, resize_shape)

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        print("Number of training samples: {}".format(len(train_dataset)))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.cfg.ANOMALY_DATASET.SHUFFLE,
            pin_memory=True,
            drop_last=True,
            num_workers=4,
            collate_fn=self.collate_fn_wrapper(),
        )

        return train_loader

    
    def predict_dataset(self) -> MVTecDRAEMTestDataset:
        mvtec_path = os.path.join(self.cfg.ANOMALY_DATASET.DATASET_PATH.ROOT, 
            self.cfg.ANOMALY_DATASET.DATASET_PATH.MVTEC,
            self.obj_name, 
            "/test/")
        resize_shape = (self.cfg.ANOMALY_DATASET.RESIZE.HEIGHT, self.cfg.ANOMALY_DATASET.RESIZE.WIDTH)
        return MVTecDRAEMTestDataset(self.cfg, mvtec_path, get_test_transforms, resize_shape)

    def predict_dataloader(self) -> DataLoader:
        val_dataset = self.val_dataset()
        print("Number of test samples: {}".format(len(val_dataset)))
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

    # @staticmethod
    def collate_fn_wrapper(self):
        def collate_fn(batch):
            
            return {
                'image': torch.stack([i['image'] for i in batch]),
                'anomaly_mask': torch.stack([i['anomaly_mask'] for i in batch]),
                'augmented_image': torch.stack([i['augmented_image'] for i in batch]),
                'has_anomaly': [i['has_anomaly'] for i in batch],
                'idx': torch.stack([i['idx'] for i in batch])
            }
        return collate_fn


