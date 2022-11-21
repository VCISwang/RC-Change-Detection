from data.transform import crop, hflip, normalize, resize, blur, cutout

import numpy as np
import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import cv2


def get_voc_pallete(num_classes):
    n = num_classes
    pallete = [0]*(n*3)
    for j in range(0,n):
            lab = j
            pallete[j*3+0] = 0
            pallete[j*3+1] = 0
            pallete[j*3+2] = 0
            i = 0
            while (lab > 0):
                    pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                    pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                    pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                    i = i + 1
                    lab >>= 3
    return pallete


def get_normalized_vector(d):
    d /= 1e-12 + torch.max(torch.abs(d))
    return d / torch.sqrt((1e-6 + torch.sum(torch.pow(d, 2.0))))


def generate_perturbation(x):
    d = torch.normal(torch.zeros(x.size()), torch.ones(x.size()))
    d = get_normalized_vector(d)
    d.requires_grad = False
    return 20 * get_normalized_vector(d)


class SemiDataset(Dataset):
    def __init__(self, root, mode, label_percent):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.palette = get_voc_pallete(2)
        self.mode = mode
        self.root = root
        self.percnt_lbl = label_percent

        self.base_size = 256
        self.crop_size = 256
        self.flip = True
        self.scale = True
        self.image_padding = (np.array(self.MEAN) * 255.).tolist()

        # self.jitter_tf = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        self.jitter_tf_s = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(self.MEAN, self.STD)

        cv2.setNumThreads(0)

        if mode == 'semi':
            labeled_id_path = os.path.join(self.root, 'list', f"{self.percnt_lbl}_{'train_supervised'}" + ".txt")
            unlabeled_id_path = os.path.join(self.root, 'list', f"{self.percnt_lbl}_{'reliable_ids'}" + ".txt")
            unre_id_path = os.path.join(self.root, 'list', f"{self.percnt_lbl}_{'unreliable_ids'}" + ".txt")
            # unlabeled_id_path = os.path.join(self.root, 'list', f"{self.percnt_lbl}_{'train_unsupervised'}" + ".txt")
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
            with open(unre_id_path, 'r') as f:
                self.unrel_ids = f.read().splitlines()
            self.ids = \
                self.labeled_ids * math.ceil(len(self.unlabeled_ids) / len(self.labeled_ids)) \
                + self.unlabeled_ids + self.unrel_ids
            # self.ids = \
            #     self.labeled_ids * math.ceil(len(self.unlabeled_ids) / len(self.labeled_ids)) \
            #     + self.unlabeled_ids + self.unrel_ids
        else:
            if self.mode == "val":
                id_path = os.path.join(self.root, 'list', f"{self.mode}" + ".txt")
            elif self.mode == "test":
                id_path = os.path.join(self.root, 'list', f"{self.mode}" + ".txt")
            elif self.mode == "select":
                id_path = os.path.join(self.root, 'list', f"{self.percnt_lbl}_{'train_unsupervised'}" + ".txt")
            elif self.mode == "label":
                id_path = os.path.join(self.root, 'list', f"{self.percnt_lbl}_{'reliable_ids'}" + ".txt")
                # id_path = os.path.join(self.root, 'list', f"{self.percnt_lbl}_{'train_unsupervised'}" + ".txt")
            elif self.mode == "train":
                id_path = os.path.join(self.root, 'list', f"{self.percnt_lbl}_{'train_supervised'}" + ".txt")
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

    def _resize(self, image_A, image_B, label, bigger_side_to_base_size=True):
        if isinstance(self.base_size, int):
            h, w, _ = image_A.shape
            if self.scale:
                longside = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
                #longside = random.randint(int(self.base_size*0.5), int(self.base_size*1))
            else:
                longside = self.base_size

            if bigger_side_to_base_size:
                h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
            else:
                h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h < w else (int(1.0 * longside * h / w + 0.5), longside)
            image_A = np.asarray(Image.fromarray(np.uint8(image_A)).resize((w, h), Image.BICUBIC))
            image_B = np.asarray(Image.fromarray(np.uint8(image_B)).resize((w, h), Image.BICUBIC))
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
            return image_A, image_B, label

        elif (isinstance(self.base_size, list) or isinstance(self.base_size, tuple)) and len(self.base_size) == 2:
            h, w, _ = image_A.shape
            if self.scale:
                scale = random.random() * 1.5 + 0.5 # Scaling between [0.5, 2]
                h, w = int(self.base_size[0] * scale), int(self.base_size[1] * scale)
            else:
                h, w = self.base_size
            image_A = np.asarray(Image.fromarray(np.uint8(image_A)).resize((w, h), Image.BICUBIC))
            image_B = np.asarray(Image.fromarray(np.uint8(image_B)).resize((w, h), Image.BICUBIC))
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
            return image_A, image_B, label

        else:
            raise ValueError

    def _crop(self, image_A, image_B, label):
        # Padding to return the correct crop size
        if (isinstance(self.crop_size, list) or isinstance(self.crop_size, tuple)) and len(self.crop_size) == 2:
            crop_h, crop_w = self.crop_size
        elif isinstance(self.crop_size, int):
            crop_h, crop_w = self.crop_size, self.crop_size
        else:
            raise ValueError

        h, w, _ = image_A.shape
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,}
        if pad_h > 0 or pad_w > 0:
            image_A = cv2.copyMakeBorder(image_A, value=self.image_padding, **pad_kwargs)
            image_B = cv2.copyMakeBorder(image_B, value=self.image_padding, **pad_kwargs)
            label   = cv2.copyMakeBorder(label, value=0, **pad_kwargs)  # use 0 for padding

        # Cropping
        h, w, _ = image_A.shape
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        end_h = start_h + crop_h
        end_w = start_w + crop_w
        image_A = image_A[start_h:end_h, start_w:end_w]
        image_B = image_B[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]
        return image_A, image_B, label

    def _flip(self, image_A, image_B, label):
        # Random H flip
        if random.random() > 0.5:
            image_A = np.fliplr(image_A).copy()
            image_B = np.fliplr(image_B).copy()
            label = np.fliplr(label).copy()
        return image_A, image_B, label

    def __getitem__(self, item):
        image_id = self.ids[item]
        if self.mode == 'semi' and image_id in self.unrel_ids:
            if random.random() < 0.5:
                image_A_path = os.path.join(self.root, 'A', image_id)
                image_B_path = os.path.join(self.root, 'A', image_id)
            else:
                image_A_path = os.path.join(self.root, 'B', image_id)
                image_B_path = os.path.join(self.root, 'B', image_id)
        else:
            image_A_path = os.path.join(self.root, 'A', image_id)
            image_B_path = os.path.join(self.root, 'B', image_id)
        image_A = np.asarray(Image.open(image_A_path), dtype=np.float32)
        image_B = np.asarray(Image.open(image_B_path), dtype=np.float32)

        if self.mode == 'val' or self.mode == 'label' or self.mode == 'test' or self.mode == 'select':
            image_A = self.normalize(self.to_tensor(Image.fromarray(np.uint8(image_A))))
            image_B = self.normalize(self.to_tensor(Image.fromarray(np.uint8(image_B))))
            label_path = os.path.join(self.root, 'label', image_id)
            label = np.asarray(Image.open(label_path), dtype=np.int32)
            if label.ndim == 3:
                label = label[:, :, 0]

            label[label >= 1] = 1
            label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
            return image_A, image_B, label, image_id

        if self.mode == 'train' or (self.mode == 'semi' and image_id in self.labeled_ids):
            label = np.asarray(Image.open(os.path.join(self.root, 'label', image_id)), dtype=np.int32)
        elif self.mode == 'semi' and image_id in self.unrel_ids:
            label = np.asarray(Image.open(os.path.join(self.root, 'label', '100.png')), dtype=np.int32)
        else:
            # mode == 'semi' and the id corresponds to unlabeled image
            # label = np.asarray(Image.open(os.path.join(self.root, 'ux_label', image_id)), dtype=np.int32)
            label = np.asarray(Image.open(os.path.join(self.root, f"{'pseudo_label'}_{self.percnt_lbl}", image_id)), dtype=np.int32)

        # basic augmentation on all training images
        h, w, _ = image_A.shape

        if self.base_size is not None:
            image_A, image_B, label = self._resize(image_A, image_B, label)

        if self.crop_size is not None:
            image_A, image_B, label = self._crop(image_A, image_B, label)

        if self.flip:
            image_A, image_B, label = self._flip(image_A, image_B, label)

        image_A = Image.fromarray(np.uint8(image_A))
        image_B = Image.fromarray(np.uint8(image_B))

        # strong augmentation on unlabeled images
        if self.mode == 'semi' and image_id in self.unlabeled_ids:
            if random.random() < 0.8:
                image_A = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_A)
                image_B = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_B)
            image_A = transforms.RandomGrayscale(p=0.2)(image_A)
            image_B = transforms.RandomGrayscale(p=0.2)(image_B)
            image_A = blur(image_A, p=0.5)
            image_B = blur(image_B, p=0.5)
            image_A, image_B, label = cutout(image_A, image_B, label, p=0.5)

        image_A = self.normalize(self.to_tensor(image_A))
        image_B = self.normalize(self.to_tensor(image_B))

        if label.ndim == 3:
            label = label[:, :, 0]
        label[label >= 1] = 1
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        return image_A, image_B, label, image_id

    def __len__(self):
        return len(self.ids)
