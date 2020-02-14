import os
from abc import ABC, abstractmethod
from torchvision.datasets import ImageFolder

from ssl_framework.config.default import cfg


class AbstractDataset(ABC):
    def __init__(self, split):
        super(AbstractDataset, self).__init__()

        self.split = split
        self.eval_mode = cfg.MODEL.FEATURE_EVAL_MODE
        self.transform = self.generate_transform()

        if self.split == 'train':
            self.root = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.TRAIN_DIR)
        else:
            self.root = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.VAL_DIR)

        self.dataset = ImageFolder(self.root, transform=self.transform)

    @abstractmethod
    def generate_transform(self):
        """generate aug transforms based on provided configuration"""
        pass