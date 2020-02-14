import argparse
import torch
import torch.nn as nn
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

class GenericSSLTrainer(pl.LightningModule):
    def __init__(self):
        super(GenericSSLTrainer, self).__init__()

        self.model = MODELS[cfg.MODEL.NAME]()

        if cfg.TRAINER.GPUS == -1:
            self.distributed = len(cfg.TRAINER.GPU_IDS) > 1
        else:
            self.distributed = cfg.TRAINER.GPUS > 1

    def forward(self, x):
        return self.model(x, cfg.MODEL.EVAL_FEATURES)

    def training_step(self, batch, batch_nb):
        data, target = batch
        logits = self.forward(data)[0]

        loss, _ = self.model.loss(logits=logits, target=target)
        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        data, target = batch
        logits = self.forward(data)[0]

        loss, output = self.model.loss(logits=logits, target=target)
        preds = torch.argmax(output, dim=1)
        val_acc = torch.sum(target == preds).item() / (len(target) * 1.0)

        return {
            'val_loss': loss,
            'val_acc': torch.tensor(val_acc)
        }

    def validation_end(self, outputs):
        val_loss_mean = 0
        val_acc_mean = 0

        for output in outputs:
            val_loss_mean += output['val_loss']
            val_acc_mean += output['val_acc']

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)

        tensorboard_logs = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}

        return {
            'val_loss': val_loss_mean,
            'val_acc': val_acc_mean,
            'log': tensorboard_logs,
            'progress_bar': tensorboard_logs
        }

    def configure_optimizers(self):
        opt_type = cfg.OPTIMIZER.TYPE
        scheduler_type = cfg.SCHEDULER.TYPE

        # get the opt params dict ready
        opt_params = dict(cfg.OPTIMIZER)
        opt_params = {k.lower(): v for k, v in opt_params.items()}
        opt_params['params'] = filter(
            lambda p: p.requires_grad, self.model.parameters())
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
        train_dataset = TASK_DATASET[cfg.DATASET.NAME]('train')

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
    model = GenericSSLTrainer()
    print('model built')

    print('#########################################')
    print('Model details')
    print('#########################################')
    print('Trunk:')
    print(model.model.trunk)
    print('#########################################')
    print('Heads:')
    print(model.model.heads)
    print('#########################################')

    # ------------------------
    # 2 DEFINE CALLBACKS
    # ------------------------
    checkpoint_callback = ModelCheckpoint(
        filepath=cfg.TRAINER.CHECKPOINT,
        save_best_only=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        gpus=cfg.TRAINER.GPU_IDS if cfg.TRAINER.GPUS == -1 else cfg.TRAINER.GPUS,
        early_stop_callback=None,
        checkpoint_callback=checkpoint_callback,
        max_nb_epochs=cfg.TRAINER.EPOCHS,
        distributed_backend='ddp',
        show_progress_bar=True
    )

    # ------------------------
    # 4 START TRAINING
    # ------------------------
    trainer.fit(model)