"""
Ref: https://lars76.github.io/2020/08/07/metrics-for-uncertainty-estimation.html
"""

import numpy as np
import torch 
import torch.nn.functional as F 

def nll(y_true, y_pred): 
    """
    Negative Log Likelihood 
    Args: 
        y_true: categorial format 
        y_pred: prediction (probability), 2D tensor, NOT in list format      
    """
    # return F.nll_loss(y_pred, y_true, reduce='mean')
    return F.nll_loss(torch.log(y_pred+1e-12), y_true, reduce='mean')


def brier_score(y_true, y_pred):
    """
    Brier Score  
    Args: 
        y_true: categorial format 
        y_pred: prediction (probability), 2D tensor, NOT in list format      
    """
    # return 1 + (np.sum(y_pred ** 2) - 2 * np.sum(y_pred[np.arange(y_pred.shape[0]), y_true])) / y_true.shape[0]
    return 1 + (torch.sum(y_pred ** 2) - 2 * torch.sum(y_pred[torch.arange(y_pred.shape[0]), y_true])) / y_true.shape[0]


def expected_calibration_error(y_true, y_pred, num_bins=15):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)

    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    return o / y_pred.shape[0]


def static_calibration_error(y_true, y_pred, num_bins=15):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()    
    classes = y_pred.shape[-1]

    o = 0
    for cur_class in range(classes):
        correct = (cur_class == y_true).astype(np.float32)
        prob_y = y_pred[..., cur_class]

        b = np.linspace(start=0, stop=1.0, num=num_bins)
        bins = np.digitize(prob_y, bins=b, right=True)

        for b in range(num_bins):
            mask = bins == b
            if np.any(mask):
                o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    return o / (y_pred.shape[0] * classes)
