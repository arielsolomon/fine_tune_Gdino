# custom_modules/weight_interpolation_hook.py
import torch
import copy
from mmengine.hooks import Hook
from mmdet.registry import HOOKS

@HOOKS.register_module()
class WeightInterpolationHook(Hook):
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.start_weights = None

    def before_train(self, runner):
        # Save pre-trained weights at the very beginning
        self.start_weights = copy.deepcopy(runner.model.state_dict())

    def after_train_epoch(self, runner):
        # Interpolate after every epoch
        current_weights = runner.model.state_dict()
        for name, param in current_weights.items():
            if name in self.start_weights:
                new_val = (1 - self.alpha) * self.start_weights[name].to(param.device) + \
                          (self.alpha) * param
                param.copy_(new_val)
