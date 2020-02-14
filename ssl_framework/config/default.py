import os

from yacs.config import CfgNode as CN


__C = CN()
cfg = __C

# trainer related params
__C.TRAINER = CN()
__C.TRAINER.NODES = 1
__C.TRAINER.GPUS = 4
__C.TRAINER.BACKEND = ''
__C.TRAINER.GPU_IDS = []
__C.TRAINER.EPOCHS = 0
__C.TRAINER.BATCH_SIZE = 16
__C.TRAINER.CHECKPOINT = ''
__C.TRAINER.WORKERS = 0
__C.TRAINER.LOGGER = 'default'
__C.TRAINER.LOG_DIR = ''

# optimizer based params
__C.OPTIMIZER = CN()
__C.OPTIMIZER.TYPE = 'sgd'
__C.OPTIMIZER.LR = 0.01
__C.OPTIMIZER.MOMENTUM = 0.9
__C.OPTIMIZER.WEIGHT_DECAY = 1e-4
__C.OPTIMIZER.NESTEROV = False
__C.OPTIMIZER.BETAS = (0.9, 0.999)

# scheduler based params
__C.SCHEDULER = CN()
__C.SCHEDULER.TYPE = ''
__C.SCHEDULER.GAMMA = 0.1
__C.SCHEDULER.STEP_SIZE = 10
__C.SCHEDULER.MILESTONES = []

# dataset related params
__C.DATASET = CN()
__C.DATASET.NAME = 'Imagenet'
__C.DATASET.TRANSFORMATION = ''
__C.DATASET.ROOT = ''
__C.DATASET.TRAIN_DIR = ''
__C.DATASET.VAL_DIR = ''

# pre-training task related params
__C.SSL_TASK = CN()
__C.SSL_TASK.NAME = ''
__C.SSL_TASK.ADDITIONAL_FILE = ''
__C.SSL_TASK.CLASSES = 10

# MODEL related params
__C.MODEL = CN()
__C.MODEL.NAME = ''
__C.MODEL.FEATURE_EVAL_MODE = False
__C.MODEL.EVAL_FEATURES = []

__C.MODEL.TRUNK = CN()
__C.MODEL.TRUNK.NAME = ''

__C.MODEL.HEAD = CN()
__C.MODEL.HEAD.PARAMS = []
__C.MODEL.HEAD.POOL = ''
__C.MODEL.HEAD.APPLY_BATCHNORM = True
__C.MODEL.HEAD.BATCHNORM_EPS = 1e-5
__C.MODEL.HEAD.BATCHNORM_MOMENTUM = 0.1

__C.NCE = CN()
__C.NCE.NEGATIVES = -1
__C.NCE.TEMP = 1.
__C.NCE.MOMENTUM = 0.5
__C.NCE.TYPE = ''

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return __C.clone()
