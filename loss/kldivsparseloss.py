import math
import numpy as np
from scipy.signal import argrelextrema
import torch
import torch.nn as nn

class KLDivSparseLoss(nn.Module):
    def __init__(self):
        super(KLDivSparseLoss, self).__init__()
        self.kldiv=nn.KLDivLoss()
        # TODO:Add the process

    def calc_sparse(self,input:torch.Tensor,hotnum:int):
        '''
        Calculate the sparcity and extrema number.
        '''
        extremas = argrelextrema(input.cpu().detach().numpy(),np.greater)
        sparse_ratio = torch.nonzero(input).shape[0]
        sparse_ratio = math.log(1.0*sparse_ratio/(input.numel()-sparse_ratio+1))
        extrema_ratio = math.log(abs(len(extremas[0]) - hotnum))
        extrema_ratio = 0 if extrema_ratio<1 else extrema_ratio
        return extrema_ratio+sparse_ratio


    def forward(self, input:torch.Tensor, positive:torch.Tensor, negative:torch.Tensor, hotnum:int):
        '''
        :param input:
        :param target:
        :param hotnum: must include
        :return:
        '''
        posdistance = self.kldiv(input, positive)
        negdistance = self.kldiv(input, negative)
        posneg_disratio = (posdistance+1e-6)/(negdistance+1)
        # print(posdistance,negdistance)
        # TODO: Add weights
        return posdistance + posneg_disratio + self.calc_sparse(input,hotnum)