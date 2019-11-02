import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import pdb

class encoder_input(nn.Module): # q_phi_x(z_x|x), q_phi_x(z_y|x), q_phi_x(z_d|x)
    def __init__(self):
        super().__init__()
        self.rep_dim = 64
        self.pool = nn.MaxPool2d(2, 2)
        bias = True

        # Encoder
        self.conv1 = nn.Conv2d(1, 32, 5, bias=bias, padding=2)
        # nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
        self.bn1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=bias, padding=2)
        # nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('relu'))
        self.bn2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(64 * 7 * 7, self.rep_dim, bias=bias)
        self.fc2 = nn.Linear(64 * 7 * 7, self.rep_dim, bias=bias)

    def sample(self, mu, logvar):
        # if self.train:
            # std = torch.exp(0.5*logvar)
            # noise = torch.randn_like(mu)
            # return noise.mul(std).add(mu)
        # else:
        #     return mu
        return torch.randn_like(logvar) * torch.exp(0.5*logvar) + mu

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        mu = self.fc1(x)
        #removed softplus for logvar got nan in KLD
        logvar = self.fc2(x)# F.softplus(self.fc2(x))
        x = self.sample(mu, logvar)
        #Removed softplus got nan in KLD
        return x, mu, logvar
#--------------------------Reconstruction(x)-----------------------------------------
# the last output 
class decoder_input(nn.Module): # p_theta(x|z_x,z_d,z_y)
    def __init__(self, in_dim):
        super().__init__()
        bias = True

        self.fc1 = nn.Linear(in_dim, 1024, bias = bias)
        self.bn1d = nn.BatchNorm1d(1024, eps=1e-04, affine=False)
        self.deconv1 = nn.Conv2d(256, 128*7*7, 5, bias=bias, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('relu'))
        self.bn2d = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.Conv2d(128, 256*2*2, 5, bias=bias, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('relu'))
        self.bn3d = nn.BatchNorm2d(256, eps=1e-04, affine=False)
#check this conv: this is the reconstruction of x and output should be 1 (for mnist) channel not 256 
        self.conv1 = nn.Conv2d(256, 1, 1, bias=bias) # no padding since the kernel size=1 , padding=2)

    def forward(self, x):
        x = F.relu(self.bn1d(self.fc1(x)))
        # x = F.interpolate(x) check if needs upsample before transpose.
        x = self.deconv1(x.view(-1, 256, 2, 2)).view(-1, 128, 14, 14)
        # print(x.shape)
        x = F.relu(self.bn2d(x))
        # print(x.shape)
        x = self.deconv2(x).view(-1, 256, 28, 28)
        # print(x.shape)
        x = F.relu(self.bn3d(x))#if needs upsample before transpose remove interpolate
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        return x
#-------------------------------------------------------------------------------
# p_theta_d ,p_theta_y
class encoder_dy(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        bias = True
        self.in_dim = in_dim

        self.fc1 = nn.Linear(self.in_dim, 64, bias=bias)
        self.bn1 = nn.BatchNorm1d(64, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(64, 64, bias=bias)
        self.fc3 = nn.Linear(64, 64, bias=bias)

    def sample(self, mu, sigma):
        # if self.train:
            # std = torch.exp(0.5*sigma)
            # noise = torch.randn_like(mu)
            # return noise.mul(std).add(mu)
        # else:
        #     return mu
        return torch.randn_like(sigma) * torch.exp(0.5*sigma) + mu

    def forward(self, x):
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = F.relu(x)
        # print(x.shape)
        mu = self.fc2(x)
        # print(mu.shape)
        #removed softplus for logvar got nan in KLD
        logvar = self.fc3(x) #F.softplus(self.fc3(x)) #beta and threshold
        # print(logvar.shape)
        x = self.sample(mu, logvar)
        
        return x, mu, logvar
#-------------------------------------------------------------------------------

class decoder_d(nn.Module): #q_w_d(d|z_d)
    def __init__(self, in_dim, d_dim):
        super().__init__()
        bias = True
        self.in_dim = in_dim
        self.fc1 = nn.Linear(self.in_dim, d_dim, bias=bias)

    def forward(self, x):
        x = F.relu(x)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
#-------------------------------------------------------------------------------

class decoder_y(nn.Module): #q_w_y(d|z_y)
    def __init__(self, in_dim, y_dim):
        super().__init__()
        bias = True
        self.in_dim = in_dim
        self.fc1 = nn.Linear(self.in_dim, y_dim, bias=bias)
    def forward(self, x):
        x = F.relu(x)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)