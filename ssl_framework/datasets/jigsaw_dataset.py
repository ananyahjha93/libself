"""
jigsaw tsk code taken from: https://github.com/bbrattoli/JigsawPuzzlePytorch
"""
import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

from ssl_framework.config.default import cfg
from ssl_framework.datasets.abstract_dataset import AbstractDataset
from ssl_framework.datasets.abstract_task import AbstractSSLTask


class JigsawSSLTask(AbstractSSLTask):
    def __init__(self):
        super(JigsawSSLTask, self).__init__()

        self.grayscale_prob = 0.3
        self.num_patches = 9

        self.permutation_file = cfg.SSL_TASK.ADDITIONAL_FILE
        self.num_classes = cfg.SSL_TASK.CLASSES
        self.permutations = self.get_permutations()

    def get_permutations(self):
        all_perm = np.load(self.permutation_file)
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

    def generate_transform(self):
        """generate aug transforms required for ssl task"""
        color_jitter_aug = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.ColorJitter(),
            transforms.ToTensor()])

        gray_scale_aug = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.Grayscale(num_output_channels=3),
            transforms.ColorJitter(),
            transforms.ToTensor()])

        return gray_scale_aug, color_jitter_aug

    def ssl_task(self, img):
        """definition of ssl task based transform we apply to individual images"""

        # apply grayscale transformation 30% of the time
        if np.random.random() < self.grayscale_prob:
            patch_aug = self.transform[0]
        else:
            patch_aug = self.transform[1]

        s = float(img.size[0]) / 3
        a = s / 2
        tiles = [None] * self.num_patches

        for n in range(self.num_patches):
            i = n / 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())

            tile = patch_aug(tile)

            # normalize patches independently
            m = tile.view(3, -1).mean(dim=1).numpy()
            s = tile.view(3, -1).std(dim=1).numpy()
            s[s == 0] = 1

            norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
            tile = norm(tile)
            tiles[n] = tile

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(self.num_patches)]
        data = torch.stack(data, 0)

        return data, order

class JigsawDataset(AbstractDataset, Dataset):
    def __init__(self, split):
        super(JigsawDataset, self).__init__(split=split)
        if not self.eval_mode:
            self.jigsaw_task = JigsawSSLTask()

    def __getitem__(self, index):
        img, target = self.dataset.__getitem__(index)

        if not self.eval_mode:
            img, target = self.jigsaw_task.ssl_task(img)

        return img, target

    def __len__(self):
        return len(self.dataset)

    def generate_transform(self):
        if self.eval_mode:
            return transforms.Compose([
                transforms.Resize(256, Image.BILINEAR),
                transforms.CenterCrop(256),
                transforms.ToTensor()
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256, Image.BILINEAR),
                transforms.CenterCrop(255)
            ])