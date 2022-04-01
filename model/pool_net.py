import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.transforms as T

# from model.hg_net import HourglassNet, Bottleneck
from lib.cpool import TopPool, BottomPool, LeftPool, RightPool

# pool(cnv_dim, TopPool, LeftPool, BottomPool, RightPool)
class CrossPool(nn.Module):
  def __init__(self, dim, pool1, pool2, pool3, pool4):
    super(CrossPool, self).__init__()
    self.p1_conv1 = convolution(3, dim, 128)
    self.p2_conv1 = convolution(3, dim, 128)
    self.p3_conv1 = convolution(3, dim, 128)
    self.p4_conv1 = convolution(3, dim, 128)
    self.p_conv1 = nn.Conv2d(128, dim, 3, padding=1, bias=False)
    self.p_bn1 = nn.BatchNorm2d(dim)
    self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(dim)
    self.conv2 = convolution(3, dim, dim)
    self.pool1 = pool1()
    self.pool2 = pool2()
    self.pool3 = pool3()
    self.pool4 = pool4()
  def forward(self, x):
    pool1 = T.functional.rotate(self.pool1(self.p1_conv1(T.functional.rotate(x, 45))), -45)
    pool2 = T.functional.rotate(self.pool2(self.p2_conv1(T.functional.rotate(x, 45))), -45)
    pool3 = T.functional.rotate(self.pool3(self.p3_conv1(T.functional.rotate(x, 45))), -45)
    pool4 = T.functional.rotate(self.pool4(self.p4_conv1(T.functional.rotate(x, 45))), -45)
    p_bn1 = self.p_bn1(self.p_conv1(pool1 + pool2 + pool3 + pool4))
    bn1 = self.bn1(self.conv1(x))
    out = self.conv2(F.relu(p_bn1 + bn1, inplace=True))
    return out

# pool(cnv_dim, TopPool, LeftPool, BottomPool, RightPool)
## MULTIPLY 45*
class StraightPool(nn.Module):
  def __init__(self, dim, pool1, pool2, pool3, pool4):
    super(StraightPool, self).__init__()
    self.p1_conv1 = convolution(3, dim, 128)
    self.p2_conv1 = convolution(3, dim, 128)
    self.p3_conv1 = convolution(3, dim, 128)
    self.p4_conv1 = convolution(3, dim, 128)
    self.p_conv1 = nn.Conv2d(128, dim, 3, padding=1, bias=False)
    self.p_bn1 = nn.BatchNorm2d(dim)
    self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(dim)
    self.conv2 = convolution(3, dim, dim)
    self.pool1 = pool1()
    self.pool2 = pool2()
    self.pool3 = pool3()
    self.pool4 = pool4()
  def forward(self, x):
    pool1 = self.pool1(self.p1_conv1(x))
    pool2 = self.pool2(self.p2_conv1(x))
    pool3 = self.pool3(self.p3_conv1(x))
    pool4 = self.pool4(self.p4_conv1(x))
    p_bn1 = self.p_bn1(self.p_conv1(pool1 + pool2 + pool3 + pool4))
    bn1 = self.bn1(self.conv1(x))
    out = self.conv2(F.relu(p_bn1 + bn1, inplace=True))
    return out

# pool(cnv_dim, TopPool, LeftPool)
# pool(cnv_dim, BottomPool, RightPool)
class TwoPool(nn.Module):
  def __init__(self, dim, pool1, pool2):
    super(pool, self).__init__()
    self.p1_conv1 = convolution(3, dim, 128)
    self.p2_conv1 = convolution(3, dim, 128)
    self.p_conv1 = nn.Conv2d(128, dim, 3, padding=1, bias=False)
    self.p_bn1 = nn.BatchNorm2d(dim)
    self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(dim)
    self.conv2 = convolution(3, dim, dim)
    self.pool1 = pool1()
    self.pool2 = pool2()
  def forward(self, x):
    pool1 = self.pool1(self.p1_conv1(x))
    pool2 = self.pool2(self.p2_conv1(x))
    p_bn1 = self.p_bn1(self.p_conv1(pool1 + pool2))
    bn1 = self.bn1(self.conv1(x))
    out = self.conv2(F.relu(p_bn1 + bn1, inplace=True))
    return out

# MaxPool(cnv_dim)
class MaxPool(nn.Module):
  def __init__(self, dim=256):
    super(MaxPool, self).__init__()
    self.p1_conv1 = convolution(3, dim, 128)
    self.p_conv1  = nn.Conv2d(128, dim, 3, padding=1, bias=False)
    self.p_bn1    = nn.BatchNorm2d(dim)
    self.conv1    = nn.Conv2d(dim, dim, 1, bias=False)
    self.bn1      = nn.BatchNorm2d(dim)
    self.conv2    = convolution(3, dim, dim)
    self.pool1    = nn.MaxPool2d(3, 1)
    self.up       = nn.UpsamplingBilinear2d(128)
  def forward(self, x):
    pool1 = self.pool1(self.p1_conv1(x))
    p_bn1 = self.p_bn1(self.p_conv1(self.up(pool1)))
    bn1 = self.bn1(self.conv1(x))
    out = self.conv2(F.relu(p_bn1 + bn1, inplace=True))
    return out

## k -kernelsize 
## inp_dim- input channel 
## out_dim- output channel 
class convolution(nn.Module):
  def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
    super(convolution, self).__init__()
    pad = (k - 1) // 2
    self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
    self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
    self.relu = nn.ReLU(inplace=True)
  def forward(self, x):
    conv = self.conv(x)
    bn = self.bn(conv)
    relu = self.relu(bn)
    return relu



class residual(nn.Module):
  def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
    super(residual, self).__init__()
    self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
    self.bn1 = nn.BatchNorm2d(out_dim)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
    self.bn2 = nn.BatchNorm2d(out_dim)
    self.skip = nn.Sequential(nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False), nn.BatchNorm2d(out_dim)) if stride != 1 or inp_dim != out_dim else nn.Sequential()
    self.relu = nn.ReLU(inplace=True)
  def forward(self, x):
    conv1 = self.conv1(x)
    bn1 = self.bn1(conv1)
    relu1 = self.relu1(bn1)
    conv2 = self.conv2(relu1)
    bn2 = self.bn2(conv2)
    skip = self.skip(x)
    return self.relu(bn2 + skip)



# inp_dim -> out_dim -> ... -> out_dim
def make_layer(kernel_size, inp_dim, out_dim, modules, layer, stride=1):
  layers = [layer(kernel_size, inp_dim, out_dim, stride=stride)]
  layers += [layer(kernel_size, out_dim, out_dim) for _ in range(modules - 1)]
  return nn.Sequential(*layers)



# inp_dim -> inp_dim -> ... -> inp_dim -> out_dim
def make_layer_revr(kernel_size, inp_dim, out_dim, modules, layer):
  layers = [layer(kernel_size, inp_dim, inp_dim) for _ in range(modules - 1)]
  layers.append(layer(kernel_size, inp_dim, out_dim))
  return nn.Sequential(*layers)



# key point layer
def make_kp_layer(cnv_dim, curr_dim, out_dim):
  return nn.Sequential(convolution(3, cnv_dim, curr_dim, with_bn=False),
                       nn.Conv2d(curr_dim, out_dim, (1, 1)))



class kp_module(nn.Module):
  def __init__(self, n, dims, modules):
    super(kp_module, self).__init__()
    self.n = n
    curr_modules = modules[0]
    next_modules = modules[1]
    curr_dim = dims[0]
    next_dim = dims[1]
    self.top = make_layer(3, curr_dim, curr_dim, curr_modules, layer=residual)
    self.down = nn.Sequential()
    self.low1 = make_layer(3, curr_dim, next_dim, curr_modules, layer=residual, stride=2)
    if self.n > 1:
      self.low2 = kp_module(n - 1, dims[1:], modules[1:])
    else:
      self.low2 = make_layer(3, next_dim, next_dim, next_modules, layer=residual)
    self.low3 = make_layer_revr(3, next_dim, curr_dim, curr_modules, layer=residual)
    self.up = nn.Upsample(scale_factor=2)
  def forward(self, x):
    up1 = self.top(x)  # 上支路residual
    down = self.down(x)  # 下支路downsample(并没有)
    low1 = self.low1(down)  # 下支路residual
    low2 = self.low2(low1)  # 下支路hourglass
    low3 = self.low3(low2)  # 下支路residual
    up2 = self.up(low3)  # 下支路upsample
    return up1 + up2  # 合并上下支路



class MaxPoolNet(nn.Module):
    ''''''
    def __init__(self, **kwd):
        super(MaxPoolNet, self).__init__()
        config = kwd['config']
        self.nstack = nstack = 2 
        nblk = 3  
        ncls = 5 
        cnv_dim = 256
        dims=[256, 256, 384, 384, 384, 512]
        curr_dim = dims[0]
        modules=[2,2,2,4]
        # self.hg = HourglassNet(Bottleneck, num_stacks=nstack, num_blocks=nblk, num_classes=ncls)
        self.pre = nn.Sequential(convolution(7, 3, 128, stride=2),
                             residual(3, 128, curr_dim, stride=2))
        # body
        self.kps = nn.ModuleList([kp_module(nblk, dims, modules) for _ in range(nstack)])
        # project
        self.cnvs = nn.ModuleList([convolution(3, curr_dim, cnv_dim) for _ in range(nstack)])
        # residual
        self.inters = nn.ModuleList([residual(3, curr_dim, curr_dim) for _ in range(nstack - 1)])
        self.inters_ = nn.ModuleList([nn.Sequential(nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False), nn.BatchNorm2d(curr_dim)) for _ in range(nstack - 1)])
        self.cnvs_ = nn.ModuleList([nn.Sequential(nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False), nn.BatchNorm2d(curr_dim)) for _ in range(nstack - 1)])
        # heatmap layers
        self.hmap = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, ncls) for _ in range(nstack)])
        # pool
        self.cnvs_pool = nn.ModuleList([MaxPool(cnv_dim) for _ in range(nstack)])
        # embedding out
        self.embd = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)])
        # regression layers
        self.regs = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])
        self.relu = nn.ReLU(inplace=True)
    def forward(self, inputs , point_list_, split, sz):
        inter = self.pre(inputs)
        outs = []
        for ind in range(self.nstack) :
            print(ind, self.nstack)    
            kp = self.kps[ind](inter)
            cnv = self.cnvs[ind](kp) 
            if ind < self.nstack - 1:
                print('inter')    
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
            if self.training or ind == self.nstack - 1:
                print('outs')    
                cnv_pool = self.cnvs_pool[ind](cnv)
                # get hm 
                hmap_ = self.hmap[ind](cnv_pool)
                # get kp
                embd_ = self.embd[ind](cnv_pool)
                # get coord
                regs_ = self.regs[ind](cnv_pool)
                outs.append([hmap_, embd_, regs_])
        # return outs
        return { 'type' : 'poolNet', 'op': outs, 'pnl': point_list_}



class CrossPoolNet(nn.Module):
    ''''''
    def __init__(self, **kwd):
        super(CrossPoolNet, self).__init__()
        config = kwd['config']
        self.nstack = nstack = 2 
        nblk = 3  
        ncls = 5 
        cnv_dim = 256
        dims=[256, 256, 384, 384, 384, 512]
        curr_dim = dims[0]
        modules=[2,2,2,4]
        # self.hg = HourglassNet(Bottleneck, num_stacks=nstack, num_blocks=nblk, num_classes=ncls)
        self.pre = nn.Sequential(convolution(7, 3, 128, stride=2),
                             residual(3, 128, curr_dim, stride=2))
        # body
        self.kps = nn.ModuleList([kp_module(nblk, dims, modules) for _ in range(nstack)])
        # project
        self.cnvs = nn.ModuleList([convolution(3, curr_dim, cnv_dim) for _ in range(nstack)])
        # residual
        self.inters = nn.ModuleList([residual(3, curr_dim, curr_dim) for _ in range(nstack - 1)])
        self.inters_ = nn.ModuleList([nn.Sequential(nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False), nn.BatchNorm2d(curr_dim)) for _ in range(nstack - 1)])
        self.cnvs_ = nn.ModuleList([nn.Sequential(nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False), nn.BatchNorm2d(curr_dim)) for _ in range(nstack - 1)])
        # heatmap layers
        self.hmap = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, ncls) for _ in range(nstack)])
        # pool
        self.cnvs_pool = nn.ModuleList([CrossPool(cnv_dim, TopPool, LeftPool, BottomPool, RightPool) for _ in range(nstack)])
        # embedding out
        self.embd = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)])
        # regression layers
        self.regs = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])
        self.relu = nn.ReLU(inplace=True)
    def forward(self, inputs , point_list_, split, sz):
        # kp, x_ = self.hg(inputs)
        inter = self.pre(inputs)
        outs = []
        for ind in range(self.nstack) :
            print(ind, self.nstack)    
            kp = self.kps[ind](inter)
            cnv = self.cnvs[ind](kp) 
            if ind < self.nstack - 1:
                print('inter')    
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
            if self.training or ind == self.nstack - 1:
                print('outs')    
                cnv_pool = self.cnvs_pool[ind](cnv)
                # get hm 
                hmap_ = self.hmap[ind](cnv_pool)
                # get kp
                embd_ = self.embd[ind](cnv_pool)
                # get coord
                regs_ = self.regs[ind](cnv_pool)
                outs.append([hmap_, embd_, regs_])
        # return outs
        return { 'type' : 'poolNet', 'op': outs, 'pnl': point_list_}

        
        


class StraightPoolNet(nn.Module):
    ''''''
    def __init__(self, **kwd):
        super(StraightPoolNet, self).__init__()
        config = kwd['config']
        self.nstack = nstack = 2 
        nblk = 3  
        ncls = 5 
        cnv_dim = 256
        dims=[256, 256, 384, 384, 384, 512]
        curr_dim = dims[0]
        modules=[2,2,2,4]
        # self.hg = HourglassNet(Bottleneck, num_stacks=nstack, num_blocks=nblk, num_classes=ncls)
        self.pre = nn.Sequential(convolution(7, 3, 128, stride=2),
                             residual(3, 128, curr_dim, stride=2))
        # body
        self.kps = nn.ModuleList([kp_module(nblk, dims, modules) for _ in range(nstack)])
        # project
        self.cnvs = nn.ModuleList([convolution(3, curr_dim, cnv_dim) for _ in range(nstack)])
        # residual
        self.inters = nn.ModuleList([residual(3, curr_dim, curr_dim) for _ in range(nstack - 1)])
        self.inters_ = nn.ModuleList([nn.Sequential(nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False), nn.BatchNorm2d(curr_dim)) for _ in range(nstack - 1)])
        self.cnvs_ = nn.ModuleList([nn.Sequential(nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False), nn.BatchNorm2d(curr_dim)) for _ in range(nstack - 1)])
        # heatmap layers
        self.hmap = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, ncls) for _ in range(nstack)])
        # pool
        self.cnvs_pool = nn.ModuleList([StraightPool(cnv_dim, TopPool, LeftPool, BottomPool, RightPool) for _ in range(nstack)])
        # embedding out
        self.embd = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)])
        # regression layers
        self.regs = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])
        self.relu = nn.ReLU(inplace=True)
    def forward(self, inputs , point_list_, split, sz):
        # kp, x_ = self.hg(inputs)
        inter = self.pre(inputs)
        outs = []
        for ind in range(self.nstack) :
            print(ind, self.nstack)    
            kp = self.kps[ind](inter)
            cnv = self.cnvs[ind](kp) 
            if ind < self.nstack - 1:
                print('inter')    
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
            if self.training or ind == self.nstack - 1:
                print('outs')    
                cnv_pool = self.cnvs_pool[ind](cnv)
                # get hm 
                hmap_ = self.hmap[ind](cnv_pool)
                # get kp
                embd_ = self.embd[ind](cnv_pool)
                # get coord
                regs_ = self.regs[ind](cnv_pool)
                outs.append([hmap_, embd_, regs_])
        # return outs
        return { 'type' : 'poolNet', 'op': outs, 'pnl': point_list_}

        
        

def poolnet(**kwd):
    if kwd['config'].model_mode == 'mpn':
        model = MaxPoolNet(config=kwd['config'])
    elif kwd['config'].model_mode == 'spn':
        model = StraightPoolNet(config=kwd['config'])
    elif kwd['config'].model_mode == 'cpn':
        model = CrossPoolNet(config=kwd['config'])
    return model

# m = exkp(n=3, nstack=2, dims=[256, 256, 384, 384, 384, 512], modules=[2,2,2,4],num_classes=5)


        # point_list, pt_idx = point_list_ 
        # # point_list = [_ for a in point_list_ for _ in a]
        # hm1, hm2, hm3 = x
        # maps = []
        # for hix in range(hm1.shape[0]) : 
        #     maps.append(hm1[hix])
        #     maps.append(hm2[hix])
        #     maps.append(hm3[hix])
        # maps = torch.stack(maps)
        # maps = maps.reshape(int(maps.shape[0]/3), -1, maps.shape[-2], maps.shape[-1])
        # maps = self.m4(maps)


        # nodes = []
        # for graphs in range(maps.shape[0]) :
        #     # print('\n', graphs,'in', range(maps.shape[0]))
        #     op = []
        #     pl = point_list[graphs]
        #     # print(pl)
        #     pnl = point_norm_list[graphs]
        #     # print(pnl)
        #     # print('pts in order')
        #     ze = 0
        #     for pt_, ptn_  in zip(pl, pnl) :
        #         # print(ze, pt_[0], pt_[1])
        #         ze+=1
        #         fp = maps[graphs, :,  pt_[0], pt_[1]]
        #         # print(fp.shape)
        #         fp_ = fp.new([ptn_[0], ptn_[1]])
        #         # print(fp_.shape)
        #         raw = torch.cat((fp, fp_))
        #         # print(raw.shape)
        #         n_e = self.node_emb(raw)
        #         # print(n_e.shape)
        #         op.append(n_e)                  
        #     nodes.append(torch.stack(op))
        # g_l = []


