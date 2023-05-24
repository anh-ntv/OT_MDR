import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model, backup=True):
    def _disable(module):
        # if isinstance(module, _BatchNorm) and not hasattr(module, "backup_momentum"):
        #     module.backup_momentum = module.momentum
        # if isinstance(module, _BatchNorm) and module.momentum != 0:
        #     module.backup_momentum = module.momentum
        # module.momentum = 0
        if isinstance(module, _BatchNorm):
            if not hasattr(module, "backup_momentum"):
                module.backup_momentum = module.momentum
            # print("_disable", module.momentum)
            module.momentum = 0
        # elif hasattr(module, 'momentum'):
        #     print("NOTBN:", module, module.momentum)
    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            # print("_enable", module.momentum)
            module.momentum = module.backup_momentum

    model.apply(_enable)
