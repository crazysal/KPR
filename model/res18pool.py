import torch 
import torch.nn as nn 
import numpy as np
import torchvision.transforms as T
from local_pooling_8dof import top_pool, bottom_pool, left_pool, right_pool, pos_diagonal_tensor, neg_diagonal_tensor, pos_anti_diagonal_tensor, neg_anti_diagonal_tensor
import itertools
import torchvision.models as models

# All pool summed 
class Pool255(nn.Module) :
    def __init__(self):
        super(Pool255, self).__init__()  
        TP  = top_pool
        BP  = bottom_pool
        LP  = left_pool
        RP  = right_pool      
        PD  = pos_diagonal_tensor
        PAD = pos_anti_diagonal_tensor
        ND  = neg_diagonal_tensor
        NAD = neg_anti_diagonal_tensor
        self.pools = [TP,BP,LP,RP,PD,PAD,ND,NAD]
        stuff = [0, 1, 2, 3, 4, 5, 6, 7]
        # self.pools = [TP,BP,LP,RP]
        # self.pools = [PD,PAD,ND,NAD]
        # stuff = [0, 1, 2, 3]
        # self.pools = [TP]
        # stuff = [0]
        self.pools255 = []
        for L in range(0, len(stuff)+1):
            for subset in itertools.combinations(stuff, L):
                self.pools255.append(subset)
        self.pools255 = self.pools255[1:]
    def forward(self, x):
        f = []
        self.pool_op = []
        for p in self.pools :
            # print(p) 
            self.pool_op.append(p(x))
        for p in self.pools255 : 
            k = []
            for _ in p :
                # print('\n doing \t', self.pools[_])
                k.append(self.pool_op[_])                
            k = torch.stack(k)
            k = torch.mean(k, dim=0)
            f.append(k)
        f = torch.stack(f)
        f = torch.mean(f, dim=0)
        return f

# f = []
# for p in pools255 : 
#     k = None
#     for _ in p : 
#         op = pools[_](x)
#         if k is None :
#             k = op
#         else : 
#             k+=op
#     k = k / len(p)
#     f.append(k)


# f = torch.stack(f)
# f = torch.mean(f, dim=0)


class Res18PoolNet(nn.Module) :
    def __init__(self):
        super(Res18PoolNet, self).__init__()         
        # r18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        r18 = models.resnet18()
        self.ft = nn.Sequential(*(list(r18.children())[:-2])) #first way 
        self.spool = Pool255()
        self.fc = nn.Linear(in_features=512, out_features=1000, bias=True)
    def forward(self, x):
        # print('batch', x.shape)       
        x = self.ft(x)
        # print('res ft', x.shape)       
        x = self.spool(x)
        # print('pool ft', x.shape)
        x = x.reshape(x.shape[0],x.shape[1], -1)       
        # print('rshae', x.shape)
        x = torch.mean(x, dim=-1)       
        # print('mean', x.shape)
        x = self.fc(x)
        # print('final', x.shape)       
        return  x

# r = Res18PoolNet()

# o__ = r(img)

'''
##create val folder 


f = open('LOC_val_solution.csv', 'r')
lns = f.readlines()


vd = {}
for i in range(1, len(lns)):
    k = lns[i].split(',')[1].split(' ')[0]
    v = lns[i].split(',')[0]
    if k in vd : 
            vd[k].append(v)
    else : 
            vd[k]= [v]


import shutil

for k in vd :
    v = vd[k] 
    for v_ in v : 
        dst = osp.join('./val', k, v_+'.JPEG')
        src = osp.join('/data/imagenet-darpit/valid', v_+'.JPEG')
        shutil.copyfile(src, dst)
/data/valid/ILSVRC2012_val_00048981.JPEG
'''