import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

def kld_loss(mu1, logvar1, mu2, logvar2, reduction = 'mean'):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    # loss = - sum(1 + log(sigma^2) - mu^2 - sigma^2)
    d = logvar1.shape[1]
    KLD_element = -d/2 + 0.5 * (logvar2.sum(dim=1) - logvar1.sum(dim=1) + 
                                (logvar1-logvar2).exp().sum(dim=1) + 
                                (mu2-mu1).pow(2).div(logvar2.exp()).sum(dim=1))
    # KLD_element = 0.5*(logvar2.sum(dim=1)-logvar1.sum(dim=1)-d+(logvar1-logvar2).exp().sum(dim=1)+ (2*(mu2-mu1).abs().log()-logvar2).exp().sum(dim=1))
    #KLD_element = (mu1-mu2).pow(2).div_(logvar2).add_((logvar1.exp()).div(logvar2.exp())).add_(logvar2).mul_(-1).add_(1).add_(logvar1)
    if reduction =='sum':
        return torch.sum(KLD_element)
    return torch.mean(KLD_element) # if the results does not make sence torch sum

def bce_loss(outputs, inputs, reduction = 'mean'):
    recon_loss = nn.MSELoss(reduction = reduction , size_average=False) #also we can use L1Loss instead of MSELoss() # if the results does not make sence reduction='sum'
    BCE = recon_loss(outputs, inputs)

    return BCE