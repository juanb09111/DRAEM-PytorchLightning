import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch
from torch.nn import functional as F
import cv2
import glob
import imgaug.augmenters as iaa
from .perlin import rand_perlin_2d_np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from kornia.filters.blur_pool import BlurPool2D
# from kornia.color import rgb_to_lab

class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, cfg, root_dir, transforms):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/scratch/*.png"))
        self.transforms = transforms
        self.cfg = cfg

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path, resize_shape):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if resize_shape != None:
            image = cv2.resize(image, dsize=(resize_shape[1], resize_shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, dsize=(resize_shape[1], resize_shape[0]), interpolation=cv2.INTER_NEAREST)

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        # image = np.transpose(image, (2, 0, 1))
        # mask = np.transpose(mask, (2, 0, 1))
        return image, mask
    
    def find_new_sizes(self, image_path):
        image = cv2.imread(image_path)
        sizes_arr = []
        for max_size in self.cfg.ANOMALY_DATASET.MAX_SIZES:

            f1 = max_size / image.shape[1]
            f2 = max_size / image.shape[0]
            min_val = min(f1,f2)
            if min_val < 1:
                resize_shape = (int(image.shape[0] * min_val), int(image.shape[1] * min_val))
            else:
                resize_shape = (image.shape[0], image.shape[1])
            
            #round
            n=64
            resize_shape = (max(int(64), round(resize_shape[0] / n) * n), max(int(64), round(resize_shape[1] / n) * n))
            sizes_arr = [*sizes_arr, resize_shape]
        return sizes_arr


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        resize_shapes = self.find_new_sizes(img_path)
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        
        basename = self.images[idx].split("/")[-1].split(".")[0]
        loc = "/".join(self.images[idx].split("/")[1:-1])

        imgs_obj = {}

        for i, size in enumerate(["s", "m", "l"]):
            resize_shape = resize_shapes[i]

            if base_dir == 'good':
                has_anomaly = np.array([0], dtype=np.float32)
                image, mask = self.transform_image(img_path, None, resize_shape)
            else:
                mask_path = os.path.join(dir_path, '../../ground_truth/')
                mask_path = os.path.join(mask_path, base_dir)
                mask_file_name = file_name.split(".")[0]+"_mask.png"
                mask_path = os.path.join(mask_path, mask_file_name)
                image, mask = self.transform_image(img_path, mask_path, resize_shape)
                has_anomaly = np.array([1], dtype=np.float32)


            apply_transforms = self.transforms(self.cfg)
            
            transformed = apply_transforms(
                image=image,
                masks=[mask]
            )



            bp = BlurPool2D(kernel_size=5, stride=1)
            source_img_bp = bp(torch.unsqueeze(transformed["image"], dim=0))
            # lab_image = rgb_to_lab(source_img_bp)
            imgs_obj[size] = {
                'source_img': torch.squeeze(source_img_bp, dim=0),
                'anomaly_mask': torch.permute(transformed["masks"][0], (2,0,1)),
            }

        return {
            "image_small": imgs_obj["s"]["source_img"],
            "image_medium": imgs_obj["m"]["source_img"],
            "image_large": imgs_obj["l"]["source_img"],

            "anomaly_mask_small": imgs_obj["s"]["anomaly_mask"],
            "anomaly_mask_medium": imgs_obj["m"]["anomaly_mask"],
            "anomaly_mask_large": imgs_obj["l"]["anomaly_mask"],

            'idx': torch.tensor(np.array([idx])),
            'basename': basename,
            "loc": loc,
            'has_anomaly': torch.tensor(np.array([has_anomaly]))
        }




class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, cfg, root_dir, anomaly_source_path, transforms):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))

        anomaly_folder = cfg.ANOMALY_DATASET.ANOMALY_FOLDER if cfg.ANOMALY_DATASET.ANOMALY_FOLDER != "" else "*"
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/{}/*.jpg".format(anomaly_folder)))

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

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-15, 15))])

        self.transforms = transforms
        self.cfg = cfg
        self.no_anomaly = 0

    def __len__(self):
        return len(self.image_paths)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path, resize_shape):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(resize_shape[1], resize_shape[0]))
        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_noise = rand_perlin_2d_np((resize_shape[0], resize_shape[1]), (perlin_scalex, perlin_scaley))

        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        # 50% percent of images contain anomalies
        # no_anomaly = torch.rand(1).numpy()[0]
        if self.no_anomaly > 0.5:
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

    def transform_image(self, image_path, anomaly_source_path, resize_shape):
        image = cv2.imread(image_path)
        # print("image_shape", image.shape, resize_shape)

        image = cv2.resize(image, dsize=(resize_shape[1], resize_shape[0]), interpolation=cv2.INTER_NEAREST)

        # do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        # if do_aug_orig:
        #     image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path, resize_shape)
        # augmented_image = np.transpose(augmented_image, (2, 0, 1))
        # image = np.transpose(image, (2, 0, 1))
        # anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly
    
    def find_new_sizes(self, image_path):
        image = cv2.imread(image_path)
        sizes_arr = []
        for max_size in self.cfg.ANOMALY_DATASET.MAX_SIZES:

            f1 = max_size / image.shape[1]
            f2 = max_size / image.shape[0]
            min_val = min(f1,f2)
            if min_val < 1:
                resize_shape = (int(image.shape[0] * min_val), int(image.shape[1] * min_val))
            else:
                resize_shape = (image.shape[0], image.shape[1])
            
            #round
            n=64
            resize_shape = (max(int(64), round(resize_shape[0] / n) * n), max(int(64), round(resize_shape[1] / n) * n))
            sizes_arr = [*sizes_arr, resize_shape]
        return sizes_arr

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        
        basename = self.image_paths[idx].split("/")[-1].split(".")[0]
        loc = "/".join(self.image_paths[idx].split("/")[1:-1])

        imgs_obj = {}
        resize_shapes = self.find_new_sizes(self.image_paths[idx])

        self.no_anomaly = torch.rand(1).numpy()[0]
        for i, size in enumerate(["s", "m", "l"]):
            resize_shape = resize_shapes[i]
            image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx], self.anomaly_source_paths[anomaly_source_idx], resize_shape)
            


            
            apply_transforms = self.transforms(self.cfg)

            transformed = apply_transforms(
                image=image,
                masks=[anomaly_mask]
            )

            transformed_augmented = apply_transforms(image=augmented_image) 
            
            bp = BlurPool2D(kernel_size=5, stride=1)
            source_img = bp(torch.unsqueeze(transformed["image"], dim=0))
            # source_img = rgb_to_lab(source_img)

            augmented_image = bp(torch.unsqueeze(transformed_augmented["image"], dim=0))
            # augmented_image = rgb_to_lab(augmented_image)

            imgs_obj[size] = {'source_img': torch.squeeze(source_img, dim=0),
                "anomaly_mask": torch.permute(transformed["masks"][-1], (2,0,1)),
                'augmented_image': torch.squeeze(augmented_image, dim=0)
            }
        
        return {
            "image_small": imgs_obj["s"]["source_img"],
            "image_medium": imgs_obj["m"]["source_img"],
            "image_large": imgs_obj["l"]["source_img"],

            "anomaly_mask_small": imgs_obj["s"]["anomaly_mask"],
            "anomaly_mask_medium": imgs_obj["m"]["anomaly_mask"],
            "anomaly_mask_large": imgs_obj["l"]["anomaly_mask"],

            "augmented_image_small": imgs_obj["s"]["augmented_image"],
            "augmented_image_medium": imgs_obj["m"]["augmented_image"],
            "augmented_image_large": imgs_obj["l"]["augmented_image"],

            'has_anomaly': torch.tensor(np.array([has_anomaly])), 
            'idx': torch.tensor(np.array([idx])),
            "loc": loc,
            "basename": basename
        }




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

    image_small = sample["image_small"]
    image_medium = sample["image_medium"]
    image_large = sample["image_large"]


    augmented_image_small = sample["augmented_image_small"]
    augmented_image_medium = sample["augmented_image_medium"]
    augmented_image_large = sample["augmented_image_large"]
    # print(sample["loc"], sample["basename"])

    preview_loc = os.path.join("datasets/preview/", sample["loc"])
    p = Path(preview_loc)
    p.mkdir(parents=True, exist_ok=True)

    save_image(image_small, os.path.join(preview_loc, sample["basename"]+"_small.png"))
    save_image(image_medium, os.path.join(preview_loc, sample["basename"]+"_medium.png"))
    save_image(image_large, os.path.join(preview_loc, sample["basename"]+"_large.png"))


    save_image(augmented_image_small, os.path.join(preview_loc, sample["basename"]+"_augmented_small.png"))
    save_image(augmented_image_medium, os.path.join(preview_loc, sample["basename"]+"_augmented_medium.png"))
    save_image(augmented_image_large, os.path.join(preview_loc, sample["basename"]+"_augmented_large.png"))

    # Visualize Semantic
    for sem in ["anomaly_mask"]:
        for s in ["_small", "_medium", "_large"]:
            im = torch.squeeze(sample[sem+s])
            plt.imshow(im)
            plt.savefig(os.path.join(preview_loc, sample["basename"]+"_anomaly_mask{}.png".format(s)))
    


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
            
            dataset_test = MVTecDRAEMTrainDataset(self.cfg, root_dir, anomaly_source_path, get_train_transforms)

            for i in range(10):
                test_dataset(cfg, dataset_test, i)


    def train_dataset(self) -> MVTecDRAEMTrainDataset:
        root_dir = os.path.join(self.cfg.ANOMALY_DATASET.DATASET_PATH.ROOT, 
            self.cfg.ANOMALY_DATASET.DATASET_PATH.MVTEC, 
            self.obj_name, "train/good/")
        anomaly_source_path = os.path.join(self.cfg.ANOMALY_DATASET.DATASET_PATH.ROOT, 
            self.cfg.ANOMALY_DATASET.DATASET_PATH.ANOMALY_SOURCE)
        
        return MVTecDRAEMTrainDataset(self.cfg, root_dir, anomaly_source_path, get_train_transforms)

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
            "test/")
        return MVTecDRAEMTestDataset(self.cfg, mvtec_path, get_test_transforms)

    def predict_dataloader(self) -> DataLoader:
        predict_dataset = self.predict_dataset()
        print("Number of test samples: {}".format(len(predict_dataset)))
        predict_loader = torch.utils.data.DataLoader(
            predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=4,
            collate_fn=self.collate_fn_wrapper(),
        )

        return predict_loader

    # @staticmethod
    def collate_fn_wrapper(self):
        def collate_fn(batch):
            max_width = 0
            max_height = 0

            for s in batch:

                for im in [s["image_small"], s["image_medium"], s["image_large"]]:

                    sample_image_shape = im.shape
                    max_height = max_height if max_height > sample_image_shape[-2] else sample_image_shape[-2]
                    max_width = max_width if max_width > sample_image_shape[-1] else sample_image_shape[-1]

          
            images_list = []
            anomaly_mask_list = []
            augmented_image_list = []
            has_anomaly_list = []
            idx_list = []
            sizes_arr = ["small", "medium", "large"]

            for idx, i in enumerate(batch):

                for s in sizes_arr:

                    batch[idx]["original_size"] = torch.tensor([batch[idx]["image_"+s].shape])
                    padding_right = max_width - batch[idx]["image_"+s].shape[-1]
                    padding_bottom = max_height - batch[idx]["image_"+s].shape[-2]
                    p2d = (0, padding_right, 0, padding_bottom)
                    image =  F.pad(batch[idx]["image_"+s], p2d, "constant", 0)
                    anomaly_mask =  F.pad(batch[idx]["anomaly_mask_"+s], p2d, "constant", 0)
                    if "augmented_image_"+s in batch[idx].keys():
                        augmented_image =  F.pad(batch[idx]["augmented_image_"+s], p2d, "constant", 0)
                        augmented_image_list.append(augmented_image)
                    
                    images_list.append(image)
                    anomaly_mask_list.append(anomaly_mask)
                    has_anomaly_list.append(batch[idx]["has_anomaly"])
                    idx_list.append(batch[idx]["idx"])

            return {
                'image': torch.stack(images_list),
                'anomaly_mask': torch.stack(anomaly_mask_list),
                'augmented_image': torch.stack(augmented_image_list) if len(augmented_image_list) > 0 else torch.stack([torch.tensor(0) for i in range(len(images_list)) ]),
                'has_anomaly': has_anomaly_list,
                'idx': torch.stack(idx_list)
            }
        return collate_fn


