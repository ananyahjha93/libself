import torch.optim.lr_scheduler as scheduler

SCHEDULERS = {'step': scheduler.StepLR,
              'multi_step': scheduler.MultiStepLR}