# 这个深度映射网络需满足：
# 1. 尺度不变性
# 2. 同圆厚度不变性

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model=128, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(4)])  # clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def __attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        # nbatches = query.size(0)


        # query, key, value = \
        #     [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #      for l, x in zip(self.linears, (query, key, value))] #3D
        query, key, value = \
            [l(x) for l, x in zip(self.linears, (query, key, value))]   #4D

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.__attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # print(x.shape)

        # # 3) "Concat" using a view and apply a final linear.
        # x = x.transpose(1, 2).contiguous() \
        #     .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class embednet(nn.Module):
    def __init__(self,h=8, d_model=128): #,o_dim=16
        super(embednet, self).__init__()
        self.d_model = d_model
        self.attn = MultiHeadedAttention(h,self.d_model)
        # self.lut = nn.Embedding(d_model,o_dim)

    def forward(self,x):
        # return self.lut(self.attn(x,x,x,None))
        x = F.interpolate(x, size=(self.d_model, self.d_model), mode='bilinear')
        return self.attn(x,x,x,None)


# Some test codes
if __name__ == "__main__":
    import numpy as np
    import cv2
    from PIL import Image

    from visdom import Visdom
    viz=Visdom()
    assert viz.check_connection()


    model=embednet().cuda()
    print(torch.load("test.pth"))
    model.load_state_dict(torch.load("test.pth"))  # ~800kB
    model.eval()

    # inputs=torch.rand(2,3,256,256).cuda()
    image=np.zeros((512, 512, 3), np.uint8)
    cv2.circle(image, (100, 120), 30, (1, 1, 1), 2)

    inputs = torch.from_numpy(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    outputs=model(inputs.unsqueeze(0))
    print(outputs.shape)
    # torch.save(model.state_dict(),"test.pth") # ~800kB

    viz.surf(X=inputs[0][0],opts=dict(colormap='Hot'))
    viz.surf(X=inputs[0][1],opts=dict(colormap='Hot'))
    viz.surf(X=inputs[0][2],opts=dict(colormap='Hot'))
    viz.surf(X=outputs[0],opts=dict(colormap='Hot'))


    # print(inputs)
    # print(outputs)

