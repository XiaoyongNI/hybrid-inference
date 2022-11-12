import torch.nn.functional as F
import numpy as np


def mse_arr_loss(outputs, gt, window=10000,Test_Loss_OnlyP=False):
    loss = 0
    outputs = outputs[-window:]
    weights = list(np.linspace(0, 1, len(outputs) + 1))[1:]
    for output, w in zip(outputs, weights):
        if Test_Loss_OnlyP:#only calculate loss on position
            loss += F.mse_loss(output[:,:,0], gt[:,:,0]) * w
        else:
            loss += F.mse_loss(output, gt) * w
    return 2 * loss/len(outputs)
