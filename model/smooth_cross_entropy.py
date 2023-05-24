import torch
import torch.nn as nn
import torch.nn.functional as F


# def smooth_crossentropy(pred, gold, smoothing=0.1):
#     n_class = pred.size(1)
#
#     one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
#     one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
#     log_prob = F.log_softmax(pred, dim=1)
#
#     return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

def smooth_crossentropy(pred, gold, smoothing=0.1, islogits=True):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)

    if islogits:
        log_prob = F.log_softmax(pred, dim=1)
    else:
        log_prob = torch.log(pred)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)


def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""

    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)
