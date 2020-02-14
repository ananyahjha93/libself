import argparse
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data.distributed import DistributedSampler

from ssl_framework.config.default import cfg
from ssl_framework.models import MODELS
from ssl_framework.optimizers import OPTIMIZERS
from ssl_framework.schedulers import SCHEDULERS
from ssl_framework.datasets import TASK_DATASET

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser(description='LIBSELF')

parser.add_argument('--config', type=str, default=None,
                    help="config file for experiment")
parser.add_argument('--local_rank', type=int, default=0, help="")

args = parser.parse_args()

# freeze config for experiment
cfg.merge_from_file(args.config)
cfg.freeze()


class GenericSSLEvaluation(pl.LightningModule):
    def __init__(self):
        super(GenericSSLEvaluation, self).__init__()

        self.model = MODELS[cfg.MODEL.NAME]()

        if cfg.TRAINER.GPUS == -1:
            self.distributed = len(cfg.TRAINER.GPU_IDS) > 1
        else:
            self.distributed = cfg.TRAINER.GPUS > 1

    def forward(self, x):
        return self.model(x, cfg.MODEL.EVAL_FEATURES)

    def training_step(self, batch, batch_nb):
        self.model.trunk.eval()

        data, target = batch
        out_feats = self.forward(data)

        tensorboard_logs = {}
        total_loss = 0.
        for idx, feat in enumerate(out_feats):
            loss, _ = self.model.loss(logits=feat, target=target)
            tensorboard_logs['head{}_loss'.format(idx)] = loss
            total_loss += loss

        return {'loss': total_loss / len(out_feats), 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        data, target = batch
        out_feats = self.forward(data)

        summary = {}
        for idx, feat in enumerate(out_feats):
            loss, output = self.model.loss(logits=feat, target=target)
            summary['head{}_val_loss'.format(idx)] = loss

            preds = torch.argmax(output, dim=1)
            val_acc = torch.sum(target == preds).item() / (len(target) * 1.0)
            summary['head{}_val_acc'.format(idx)] = torch.tensor(val_acc)

        return summary

    def validation_end(self, outputs):
        summary = {}

        for key, _ in outputs[0].items():
            summary[key] = 0.

        for output in outputs:
            for key, val in output.items():
                summary[key] += val

        for key, _ in summary.items():
            summary[key] /= len(outputs)

        tensorboard_logs = deepcopy(summary)
        summary['log'] = tensorboard_logs
        summary['progress_bar'] = tensorboard_logs

        return summary

    def weight_decay(self):
        decay = []
        no_decay = []

        for name, param in self.model.named_parameters():
            if param.requires_grad and 'trunk' not in name:
                if 'bias' in name or 'bn' in name or len(param.shape) == 1:
                    no_decay.append(param)
                else:
                    decay.append(param)

        return [{'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': cfg.OPTIMIZER.WEIGHT_DECAY}]

    def configure_optimizers(self):
        opt_type = cfg.OPTIMIZER.TYPE
        scheduler_type = cfg.SCHEDULER.TYPE

        # get the opt params dict ready
        opt_params = dict(cfg.OPTIMIZER)
        opt_params = {k.lower(): v for k, v in opt_params.items()}

        opt_params['params'] = self.weight_decay()
        opt_params = {
            k: v for k, v in opt_params.items() \
            if k in OPTIMIZERS[opt_type].__init__.__code__.co_varnames}

        optimizer = OPTIMIZERS[opt_type](**opt_params)

        # get scheduler params dict ready
        scheduler_params = dict(cfg.SCHEDULER)
        scheduler_params = {k.lower(): v for k, v in scheduler_params.items()}
        scheduler_params['optimizer'] = optimizer
        scheduler_params = {
            k: v for k, v in scheduler_params.items() \
            if k in SCHEDULERS[scheduler_type].__init__.__code__.co_varnames}

        scheduler = SCHEDULERS[scheduler_type](**scheduler_params)

        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        train_dataset = dataset = TASK_DATASET[cfg.DATASET.NAME]('train')

        if self.distributed:
            train_sampler = DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.TRAINER.BATCH_SIZE,
            shuffle=False if self.distributed else True,
            num_workers=cfg.TRAINER.WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler
        )

        return train_loader

    @pl.data_loader
    def val_dataloader(self):
        val_dataset = TASK_DATASET[cfg.DATASET.NAME]('val')

        if self.distributed:
            val_sampler = DistributedSampler(val_dataset)
        else:
            val_sampler = None

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.TRAINER.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TRAINER.WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=val_sampler
        )

        return val_loader

if __name__ == "__main__":
    print('#########################################')
    print('Experiment configuration')
    print('#########################################')
    print(cfg)
    print('#########################################')

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    print('loading model...')
    lightning_module = GenericSSLEvaluation()
    print('model built')

    print('#########################################')
    print('Model details')
    print('#########################################')
    print('Trunk:')
    print(lightning_module.model.trunk)
    print('#########################################')
    print('Heads:')
    print(lightning_module.model.heads)
    print('#########################################')

    # ------------------------
    # 2 RESTORE TRUNK WEIGHTS
    # ------------------------
    state_dict = torch.load(
        cfg.TRAINER.CHECKPOINT,
        map_location=lambda storage, loc: storage)['state_dict']

    trunk_state_dict = {}
    for key, val in state_dict.items():
        if 'trunk' in key:
            trunk_state_dict[key.replace('model.trunk.', '')] = val

    if len(list(trunk_state_dict.keys())) == 0:
        print('no params with key trunk, loading original state_dict...')
        trunk_state_dict = state_dict

    lightning_module.model.trunk.load_state_dict(trunk_state_dict)
    print('weights from pre-trained trunk restored...')
    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        nb_gpu_nodes=cfg.TRAINER.NODES,
        gpus=cfg.TRAINER.GPU_IDS if cfg.TRAINER.GPUS == -1 else cfg.TRAINER.GPUS,
        early_stop_callback=None,
        max_nb_epochs=cfg.TRAINER.EPOCHS,
        distributed_backend='ddp',
        show_progress_bar=True
    )

    # ------------------------
    # 4 START TRAINING
    # ------------------------
    trainer.fit(lightning_module)