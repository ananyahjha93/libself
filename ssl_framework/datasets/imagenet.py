import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

from ssl_framework.config.default import cfg
from ssl_framework.datasets.abstract_dataset import AbstractDataset
from ssl_framework.datasets.abstract_task import AbstractSSLTask

from ssl_framework.transforms import Augmentation, fa_resnet50_rimagenet
from ssl_framework.transforms import _IMAGENET_PCA, Lighting


class Imagenet(AbstractDataset, Dataset):
    def __init__(self, split):
        super(Imagenet, self).__init__(split=split)

    def __getitem__(self, index):
        img, target = self.dataset.__getitem__(index)

        if cfg.MODEL.FEATURE_EVAL_MODE:
            return img, target
        else:
            return img, target, index

    def __len__(self):
        return len(self.dataset)

    def generate_transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if self.eval_mode:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            train_transforms = None

            if cfg.DATASET.TRANSFORMATION == 'fast_auto_augment':
                train_transforms = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.), interpolation=Image.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                    ),
                    transforms.ToTensor(),
                    Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
                    normalize
                ])

                train_transforms.transforms.insert(0, Augmentation(fa_resnet50_rimagenet()))
            elif cfg.DATASET.TRANSFORMATION == 'default':
                train_transforms = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.2,1.)),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])

            assert train_transforms is not None,\
                   "no transformation selected for training set"

            return train_transforms