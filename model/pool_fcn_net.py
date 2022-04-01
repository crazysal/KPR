import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.transforms as T

# from model.hg_net import HourglassNet, Bottleneck
from lib.cpool import TopPool, BottomPool, LeftPool, RightPool

# pool(cnv_dim, TopPool, LeftPool, BottomPool, RightPool)
class CrossPool(nn.Module):
  def __init__(self, dim, pool1, pool2, pool3, pool4, shp=128):
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
    self.up       = nn.UpsamplingBilinear2d(shp)
  def forward(self, x):
    pool1 = T.functional.rotate(self.pool1(self.p1_conv1(T.functional.rotate(x, 45))), -45)
    pool2 = T.functional.rotate(self.pool2(self.p2_conv1(T.functional.rotate(x, 45))), -45)
    pool3 = T.functional.rotate(self.pool3(self.p3_conv1(T.functional.rotate(x, 45))), -45)
    pool4 = T.functional.rotate(self.pool4(self.p4_conv1(T.functional.rotate(x, 45))), -45)
    # p_bn1 = self.p_bn1(self.p_conv1(pool1 + pool2 + pool3 + pool4))
    p_bn1 = self.p_bn1(self.p_conv1(self.up(pool1) + self.up(pool2) + self.up(pool3) + self.up(pool4)))
    # bn1 = self.bn1(self.conv1(x))
    bn1 = self.bn1(self.conv1(self.up(x)))
    out = self.conv2(F.relu(p_bn1 + bn1, inplace=True))
    return out

# pool(cnv_dim, TopPool, LeftPool, BottomPool, RightPool)
## MULTIPLY 45*
class StraightPool(nn.Module):
  def __init__(self, dim, pool1, pool2, pool3, pool4, shp=128):
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
    self.up       = nn.UpsamplingBilinear2d(shp)
  def forward(self, x):
    pool1 = self.pool1(self.p1_conv1(x))
    pool2 = self.pool2(self.p2_conv1(x))
    pool3 = self.pool3(self.p3_conv1(x))
    pool4 = self.pool4(self.p4_conv1(x))
    # p_bn1 = self.p_bn1(self.p_conv1(pool1 + pool2 + pool3 + pool4))
    # bn1 = self.bn1(self.conv1(x))
    p_bn1 = self.p_bn1(self.p_conv1(self.up(pool1) + self.up(pool2) + self.up(pool3) + self.up(pool4)))
    bn1 = self.bn1(self.conv1(self.up(x)))
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
  def __init__(self, dim=256, shp=128):
    super(MaxPool, self).__init__()
    self.p1_conv1 = convolution(3, dim, 128)
    self.p_conv1  = nn.Conv2d(128, dim, 3, padding=1, bias=False)
    self.p_bn1    = nn.BatchNorm2d(dim)
    self.conv1    = nn.Conv2d(dim, dim, 1, bias=False)
    self.bn1      = nn.BatchNorm2d(dim)
    self.conv2    = convolution(3, dim, dim)
    self.pool1    = nn.MaxPool2d(3, 1)
    self.up       = nn.UpsamplingBilinear2d(shp)
  def forward(self, x):
    pool1 = self.pool1(self.p1_conv1(x))
    p_bn1 = self.p_bn1(self.p_conv1(self.up(pool1)))
    # bn1 = self.bn1(self.conv1(x))
    bn1 = self.bn1(self.conv1(self.up(x)))
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



class StraightPoolNet(nn.Module) :
    def __init__(self):
        super(StraightPoolNet, self).__init__()  
        out_dim = 2048
        curr_dim = 512
        conv_dim = 256
        ncls = 5
        fcn = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
        self.ft = nn.Sequential(*(list(fcn.children())[:-2])) #first way 
        self.flt = convolution(3, out_dim, curr_dim)
        self.spool = StraightPool(curr_dim, TopPool, LeftPool, BottomPool, RightPool)
        self.hm = make_kp_layer(curr_dim, conv_dim, ncls)
        self.emb = make_kp_layer(curr_dim, conv_dim, 1)
        self.reg = make_kp_layer(curr_dim, conv_dim, 2)
    def forward(self, x, point_list_, split, sz):
        input_shape = x.shape[2]
        # print(x.shape)
        x = self.flt(self.ft(x)['out'])
        # print(x.shape)
        # x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False) 
        x = self.spool(x)
        # print(x.shape)
        hm = self.hm(x)
        # print(hm.shape)
        emb = self.emb(x)
        # print(emb.shape)
        reg = self.reg(x)
        # print(emb.shape)
        return  { 'type' : 'poolfcnNet', 'op': (hm, emb, reg), 'pnl': point_list_}
        # return hm, emb

class CrossPoolNet(nn.Module) :
    def __init__(self):
        super(CrossPoolNet, self).__init__()  
        out_dim = 2048
        curr_dim = 512
        conv_dim = 256
        ncls = 5
        fcn = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
        self.ft = nn.Sequential(*(list(fcn.children())[:-2])) #first way 
        self.flt = convolution(3, out_dim, curr_dim)
        self.cpool = CrossPool(curr_dim, TopPool, LeftPool, BottomPool, RightPool)
        self.hm = make_kp_layer(curr_dim, conv_dim, ncls)
        self.emb = make_kp_layer(curr_dim, conv_dim, 1)
        self.reg = make_kp_layer(curr_dim, conv_dim, 2)
    def forward(self, x, point_list_, split, sz):
        input_shape = x.shape[2]
        # print(x.shape)
        x = self.flt(self.ft(x)['out'])
        # print(x.shape)
        # x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False) 
        x = self.cpool(x)
        # print(x.shape)
        hm = self.hm(x)
        # print(hm.shape)
        emb = self.emb(x)
        # print(emb.shape)
        reg = self.reg(x)
        # print(emb.shape)
        return  { 'type' : 'poolfcnNet', 'op': (hm, emb, reg), 'pnl': point_list_}
        # return hm, emb


class MaxPoolNet(nn.Module) :
    def __init__(self):
        super(MaxPoolNet, self).__init__()  
        out_dim = 2048
        curr_dim = 512
        conv_dim = 256
        ncls = 5
        fcn = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
        self.ft = nn.Sequential(*(list(fcn.children())[:-2])) #first way 
        self.flt = convolution(3, out_dim, curr_dim)
        self.mpool = MaxPool(curr_dim)
        self.hm = make_kp_layer(curr_dim, conv_dim, ncls)
        self.emb = make_kp_layer(curr_dim, conv_dim, 1)
        self.reg = make_kp_layer(curr_dim, conv_dim, 2)
    def forward(self, x, point_list_, split, sz):
        input_shape = x.shape[2]
        # print(x.shape)
        x = self.flt(self.ft(x)['out'])
        # print(x.shape)
        # x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False) 
        x = self.mpool(x)
        # print(x.shape)
        hm = self.hm(x)
        # print(hm.shape)
        emb = self.emb(x)
        # print(emb.shape)
        reg = self.reg(x)
        # print(emb.shape)
        return  { 'type' : 'poolfcnNet', 'op': (hm, emb, reg), 'pnl': point_list_}
        # return hm, emb


class Mdl(nn.Module) :
    def __init__(self):
        super(Mdl, self).__init__()  
        out_dim = 2048
        curr_dim = 512
        conv_dim = 256
        ncls = 5
        fcn = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
        self.ft = nn.Sequential(*(list(fcn.children())[:-2])) #first way 
        self.flt = convolution(3, out_dim, curr_dim)
        self.base = Base(curr_dim)
        self.hm = make_kp_layer(curr_dim, conv_dim, ncls)
        self.emb = make_kp_layer(curr_dim, conv_dim, 1)
        self.reg = make_kp_layer(curr_dim, conv_dim, 2)
    def forward(self, x, point_list_, split, sz):
        input_shape = x.shape[2]
        # print(x.shape)
        x = self.flt(self.ft(x)['out'])
        # print(x.shape)
        x = self.base(x)
        # x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False) 
        hm = self.hm(x)
        # print(hm.shape)
        emb = self.emb(x)
        # print(hm.shape)
        reg = self.reg(x)
        # print(emb.shape)
        return  { 'type' : 'poolfcnNet', 'op': (hm, emb, reg), 'pnl': point_list_}
        # return hm, emb


# Base(cnv_dim)
class Base(nn.Module):
    def __init__(self, dim=256, shp=128):
        super(Base, self).__init__()
        self.p1_conv1 = convolution(3, dim, int(dim/2))
        self.p_conv1  = nn.Conv2d(int(dim/2), dim, 3, padding=1, bias=False)
        self.p_bn1    = nn.BatchNorm2d(dim)
        self.conv1    = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn1      = nn.BatchNorm2d(dim)
        self.conv2    = convolution(3, dim, dim)
        self.pool1    = nn.MaxPool2d(3, 1)
        self.up       = nn.UpsamplingBilinear2d(shp)
    def forward(self, x):
        pool1 = self.p1_conv1(x)
        # print('2', pool1.shape)
        pool1 = self.up(pool1)
        # print('3', pool1.shape)
        pool1 = self.p_conv1(pool1)
        # print('4', pool1.shape)
        pool1 = self.p_bn1(pool1)
        # print('5', pool1.shape)
        bn1 = self.bn1(self.conv1(self.up(x)))
        # print('6', bn1.shape)
        out = self.conv2(F.relu(pool1 + bn1, inplace=True))
        # print('7', out.shape)
        return out


def poolnet(**kwd):
    if kwd['config'].model_mode == 'nopn':
        model = Mdl()
    if kwd['config'].model_mode == 'mpn':
        model = MaxPoolNet()
    elif kwd['config'].model_mode == 'spn':
        model = StraightPoolNet()
    elif kwd['config'].model_mode == 'cpn':
        model = CrossPoolNet()
    return model
