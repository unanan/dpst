import torch
import torch.nn as nn

class KLDivSparseLoss(nn.Module):
    def __init__(self):
        super(KLDivSparseLoss, self).__init__()
        self.kldiv=nn.KLDivLoss()
        # TODO:Add the process
        self.sparse=None

    def forward(self, input:torch.Tensor, target:torch.Tensor,hotnum:torch.Tensor):
        '''
        :param input:
        :param target:
        :param hotnum: must include
        :return:
        '''
        return self.kldiv(input, target)+self.sparse(input,target)