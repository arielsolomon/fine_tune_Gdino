import torch
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
import copy

@HOOKS.register_module()
class WeightInterpolationHook(Hook):
    def __init__(self, alpha=0.5):
        self.alpha = alpha  # Lambda/Alpha: 1.0 = all new, 0.0 = all old
        self.start_weights = None

    def before_train(self, runner):
        # Save a copy of the pre-trained weights before training starts
        self.start_weights = copy.deepcopy(runner.model.state_dict())

    def after_train_epoch(self, runner):
        # The "Blending" step
        current_weights = runner.model.state_dict()
        
        for name, param in current_weights.items():
            if name in self.start_weights:
                # Interpolation formula: W_final = (1-a)W_start + (a)W_fine_tuned
                new_value = (1 - self.alpha) * self.start_weights[name].to(param.device) + \
                            (self.alpha) * param
                param.copy_(new_value)
        
        runner.logger.info(f"Weights interpolated with alpha {self.alpha}")
