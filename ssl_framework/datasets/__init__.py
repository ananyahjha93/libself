from ssl_framework.datasets.abstract_task import AbstractSSLTask
from ssl_framework.datasets.abstract_dataset import AbstractDataset

from ssl_framework.datasets.jigsaw_dataset import JigsawDataset
from ssl_framework.datasets.imagenet import Imagenet

TASK_DATASET = {'jigsaw_dataset': JigsawDataset,
                'imagenet': Imagenet}