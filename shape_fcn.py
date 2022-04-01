import torch 
import torchvision.models as models
from torch.nn import functional as F
import torchvision.transforms as T
from scipy.ndimage import gaussian_filter
import PIL 
from PIL import Image, ImageDraw
import argparse
import os
import gc
import os.path as osp
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import copy
from sklearn.cluster import KMeans
        
# from skimage.metrics import (adapted_rand_error,
                            #   variation_of_information)
from lib.cpool import TopPool, BottomPool, LeftPool, RightPool

from measures import compute_mae, compute_pre_rec
torch.manual_seed(17)
np.random.seed(19680801)
# resnet18 = models.resnet18(pretrained=True)
# features = nn.Sequential(*(list(resnet18.children())[:-2])) #first way 

# fcn = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
# features = nn.Sequential(*(list(fcn.children())[:-2])) #first way 
def _tranpose_and_gather_feature(feature, ind):
    # print('1', feature.shape, ind.shape)
    feature = feature.permute(1, 2, 0).contiguous()  # [C, H, W] => [H, W, C]
    # print('2', feature.shape, ind.shape)
    feature = feature.view( -1, feature.size(2))  # [H, W, C] => [H x W, C]
    # print('3', feature.shape, ind.shape)
    ind = ind[:, None].expand(ind.shape[0], feature.shape[-1])  # [num_obj] => [num_obj, C]
    # print('4', feature.shape, ind.shape)
    feature = feature.gather(0, ind)  # [H x W, C] => [num_obj, C]
    # print('5', feature.shape, ind.shape)
    return feature


def point_to_index(points, pr=224) :
    # print(points)
    coord_range = {}
    key = 0
    for x_ in range(pr) :
        for y_ in range(pr) :
            coord_range.update({(x_, y_):key})
            key+=1
    # print('coord_range', coord_range)
    ind = []
    for ln in points : 
        l = [coord_range[(ln[0].item()),(ln[1].item())]]
        ind.append(l)
    return torch.tensor(ind ).long().cuda()


class Shp(Dataset) :
    def __init__(self, **kwd):
        config = kwd['config']
        self.split=config['split']
        self.im_dir = '/home/sahmed9/reps/KeyPointRelations/data/shapes/images/'
        self.im = os.listdir(self.im_dir)
        self.an_dir = '/home/sahmed9/reps/KeyPointRelations/data/shapes/annotations/'
        self.files= {
            'train': self.im[:7500],
            'test' : self.im[7500:9000],
            'val'  : self.im[9000:]
        }
        self.pp1 = T.Compose([T.ToTensor(),T.Normalize(mean=0.5, std=0.5)])

    def __len__(self):
        return len(self.files[self.split])
    def __getitem__(self, index):
        im_ = osp.join(self.im_dir, self.files[self.split][index])
        an_ = osp.join(self.an_dir, self.files[self.split][index][:-3]+'npy')
        img = Image.open(im_)
        img = self.pp1(img)
        ann = list(np.load(an_, allow_pickle=True))
        pc  = torch.tensor([len(_) for _ in ann])
        ann = ann[0]+ann[1]+ann[2]+ann[3]
        ann = torch.tensor(ann).long()
        canvas = torch.zeros(img.shape[-1], img.shape[-1])
        for _ in ann :
            canvas[_[0], _[1]] +=1
        canvas = torch.tensor(gaussian_filter(canvas, 1)).unsqueeze(0)
        return img, canvas, ann, pc

class Mdl_s(nn.Module) :
    def __init__(self):
        super(Mdl_s, self).__init__()  
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
    def forward(self, x):
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
        return hm, emb

class Mdl_c(nn.Module) :
    def __init__(self):
        super(Mdl_c, self).__init__()  
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
    def forward(self, x):
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
        return hm, emb

class Mdl_m(nn.Module) :
    def __init__(self):
        super(Mdl_m, self).__init__()  
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
    def forward(self, x):
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
        return hm, emb


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
    def forward(self, x):
        input_shape = x.shape[2]
        print(x.shape)
        x = self.flt(self.ft(x)['out'])
        print(x.shape)
        x = self.base(x)
        # x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False) 
        hm = self.hm(x)
        print(hm.shape)
        emb = self.emb(x)
        print(emb.shape)
        return hm, emb


class Loss(nn.Module) :
    def __init__(self):
        super(Loss, self).__init__()          
        self.rec_loss = F.binary_cross_entropy_with_logits
        self.cls_loss = F.cross_entropy
        self.regr_loss = WeightedMSELoss()
        self.cls_t = 0.001
        self.alpha = 0.5
        self.beta = 0.3
        self.gamma = 0.2

    @staticmethod
    def bce_wt(lbl_bce) :
        return  torch.ones_like(lbl_bce) / 99.99  +  (1.0 - 1.0 / 99.99) * lbl_bce
        
    @staticmethod
    def mse_wt() :
        return torch.tensor([[0.01, 0.99]]).cuda()
    
    @staticmethod
    def ce_wt() :
        return torch.tensor([0.01, 0.99]).cuda()
    def forward(self, hm, emb, gt_lbl, ann, pc, spl, wt, ep, ix):
        # print('hm.shape', hm.shape)
        # print('emb.shape', emb.shape)
        # print('ann.shape', ann.shape)
        # print('pc', pc)
        pc = [tuple(list(_)) for _ in  pc.detach().cpu().numpy()]
        # print('pc', pc)
        mse_lbl = copy.deepcopy(gt_lbl/gt_lbl.max())
        mse_lbl = [1-mse_lbl, mse_lbl]
        mse_lbl = torch.cat(mse_lbl, dim=1)
        # print('mse_lbl', mse_lbl.shape)
        bce_lbl = copy.deepcopy(gt_lbl/gt_lbl.max())
        bce_lbl = torch.sigmoid(bce_lbl).squeeze(1)
        # print('bce_lbl', bce_lbl.shape)
        ce_lbl = copy.deepcopy(gt_lbl)
        ce_lbl[ce_lbl>=self.cls_t] = 1
        ce_lbl[ce_lbl<1] = 0
        ce_lbl = ce_lbl.long().squeeze(1)
        # print('ce_lbl', ce_lbl.shape)

        fm_mse = hm[:, 0:2, :, :]
        # print('fm_mse', fm_mse.shape)
        fm_bce = hm[:, 2, :, :]
        # print('fm_bce', fm_bce.shape)
        fm_ce  = torch.softmax(hm[:, 3:5, :, :],dim=1)
        # print('fm_ce', fm_ce.shape)
        l1 = self.regr_loss(fm_mse, mse_lbl, self.mse_wt()) 
        # print(l1)
        ### BCE
        l2 = self.rec_loss(fm_bce, bce_lbl, weight=self.bce_wt(bce_lbl))
        # print(l2)
        ### CE
        l3 = self.cls_loss(fm_ce, ce_lbl, weight=self.ce_wt()) 
        # print(l3)
        hm_loss = self.alpha*l1+self.beta*l2 +self.gamma*l3
        wt.add_scalar(spl +" Loss MSE", l1.item(), (ep+1)*(ix+1))
        wt.add_scalar(spl +" Loss BCE", l2.item(), (ep+1)*(ix+1))
        wt.add_scalar(spl +" Loss CE", l3.item(), (ep+1)*(ix+1)) 
        ############## 
        pll , psh  = [], []  
        for ix in range(emb.shape[0]) :
            # print(ix, 'ann[ix, :, :]', ann[ix, :, :])
            inds = point_to_index(ann[ix, :, :]).squeeze(-1)
            # print('inds', inds)
            point_ae = _tranpose_and_gather_feature(emb[ix, :, :, :], inds).squeeze(-1)
            # print('point_ae', point_ae,point_ae.shape )
            point_ae = point_ae.split(pc[ix])
            # print('point_ae', point_ae )
            pus, pul = get_push_pull_lines(point_ae)
            pll.append(pul)
            psh.append(pus)
        
        pll = sum(pll)/len(pll)
        psh = sum(psh)/len(psh)
        l0 = hm_loss
        l1 = psh
        l2 = pll
        print(spl +" Loss hm ", l0.item())
        print(spl +" Loss push ", l1.item())
        print(spl +" Loss pull ", l2.item())
        wt.add_scalar(spl +" Loss Push", l1.item(), (ep+1)*(ix+1))
        wt.add_scalar(spl +" Loss Pull", l2.item(), (ep+1)*(ix+1))
        tr_l = l0+l1+l2
        
        print(spl +" Loss ToTal", round(tr_l.item(), 5))
        print('_'*30)
        wt.add_scalar(spl +" Loss ToTal", tr_l.item(), (ep+1)*(ix+1))
        # exit()
        return tr_l


def get_push_pull_lines(lines):
    num = len(lines)
    # print('num of lines', num)
    pull, push = torch.zeros([], requires_grad=True).cuda(), torch.zeros([], requires_grad=True).cuda()
    mean_ = []
    margin = 1
    for ln in lines : 
        # print('ln.shape', ln.shape)
        line_mean = sum(ln)/ln.shape[0]
        mean_.append(line_mean)
        # print('line_mean', line_mean)
        e = [torch.pow(_ - line_mean, 2) / (ln.shape[0] + 1e-4) for _ in ln ]
        # print('e', e)
        e = sum(e)
        # print('e', e)
        pull +=e
        # print('pull', pull)

    for k, m in enumerate(mean_): 
        for j in range(num) :
            if j !=k : 
                # print('mean, linejsum', m,  lines[j].sum())
                r_ = torch.abs(m - lines[j].sum()/ (lines[j].shape[0] + 1e-4) )
                # print('r_', r_)
                d = F.relu(margin - r_)
                # print('d', d)
                d/= ((num - 1) * num + 1e-4)
                # print('d', d)
                push+= d 
                # print('push', push)
    return push, pull 



# pool(cnv_dim, TopPool, LeftPool, BottomPool, RightPool)
class CrossPool(nn.Module):
  def __init__(self, dim, pool1, pool2, pool3, pool4, shp=224):
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
  def __init__(self, dim, pool1, pool2, pool3, pool4, shp=224):
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
  def __init__(self, dim=256, shp=224):
    super(MaxPool, self).__init__()
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
    # print('1', pool1.shape)
    pool1 = self.pool1(pool1)
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

# Base(cnv_dim)
class Base(nn.Module):
  def __init__(self, dim=256, shp=224):
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



# key point layer
def make_kp_layer(cnv_dim, curr_dim, out_dim):
  return nn.Sequential(convolution(3, cnv_dim, curr_dim, with_bn=False),
                       nn.Conv2d(curr_dim, out_dim, (1, 1)))



class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = torch.sigmoid(heatmaps_pred[idx].squeeze())
            heatmap_gt = heatmaps_gt[idx].squeeze()
            loss += self.criterion(heatmap_pred.mul(target_weight[:, idx]),heatmap_gt.mul(target_weight[:, idx]))
            
        return loss / num_joints


def get_mae(hm, lbl):
    m = []
    prs = []
    rec = []
    for i in range(hm.shape[0]) : 
        h_ = torch.sigmoid(hm[i, 1, :, :]).cpu().detach().numpy()
        m_ = lbl[i, :, :, :].squeeze(0).cpu().detach().numpy()
        # print(m_)
        m_ = m_/(np.max(m_) + 1e-8)
        # print(m_)
        # print(m_, h_)
        # print(m_.shape, h_.shape)
        m.append(compute_mae(m_, h_))
        # p, r = compute_pre_rec(m_, h_)
        # prs.append(p)
        # rec.append(r)
        # print(m_.shape, h_.shape, prs[-1].shape, rec[-1].shape)
    # return m, prec 
    return m 

def get_clusters(X, nc=4):
    kmeans = KMeans(n_clusters=nc, random_state=0).fit(X)
    return kmeans.labels_

def get_simmilarity(X):

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)        
        out = self.gamma*out + x
        return out,attention

if __name__ == '__main__' :
    train()
    test()

def test():
    
    test_config = {
        'split'      :'test',
        'batch_size' : 1,
        'lr' : 1e-4,
        'epoch' : 5,
        'cache_file': 'shapes_01', 
        'root' : '/home/sahmed9/reps/KeyPointRelations/cache/',
        'use_model' : 1
    }
    tsds = Shp(config=test_config)
    tsdl = DataLoader(tsds, batch_size=test_config['batch_size'], shuffle=False)
    print('got test dl', len(vdl))

    if test_config['use_model'] == 1:
        mdl = Mdl()
    elif test_config['use_model'] == 2:
        mdl = Mdl_m()
    elif test_config['use_model'] == 3:
        mdl = Mdl_s()
    elif test_config['use_model'] == 4:
        mdl = Mdl_c()
    print('model initialized')
    print(mdl)
    total_params = 0
    for params in mdl.parameters():
        num_params = 1
        for x in params.size():
            num_params *= x
        total_params += num_params
    print('Totoal model params = ', total_params)
    mdl = mdl.cuda()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, mdl.parameters()), lr=train_config['lr'])
    print('Optim initialized')
    print(opt)
    running_accuracy = []
    with torch.no_grad() :
        mdl = mdl.eval()
        for ix, data in enumerate(iter(tsdl)) :
            gc.collect()
            image, gt_lbl, ann, pc = data
            image, gt_lbl = image.cuda(), gt_lbl.cuda()
            hm, emb = mdl(image)
            mae = get_mae(hm, gt_lbl)
            # mae, prec = get_mae(hm, gt_lbl)
            print('MAE ', sum(mae)/len(mae))            
            # print('prec ', prec)            
            for _ in mae :
                running_accuracy.append(_)
            

def train():
    train_config = {
        'split'      :'train',
        'batch_size' : 8,
        'lr' : 1e-4,
        'epoch' : 5,
        'cache_file': 'shapes_01', 
        'root' : '/home/sahmed9/reps/KeyPointRelations/cache/',
        'use_model' : 1

    }
    train_config['cache_file'] += '_'+str(train_config['use_model'])+'_'
    tds = Shp(config=train_config)
    tdl = DataLoader(tds, batch_size=train_config['batch_size'], shuffle=True)
    print('got train dl', len(tdl))
    
    val_config = {
        'split'      :'val',
        'batch_size' : 8,
        'freq' : 1
    }
    vds = Shp(config=val_config)
    vdl = DataLoader(vds, batch_size=val_config['batch_size'], shuffle=False)
    print('got val dl', len(vdl))

    test_config = {
        'split'      :'test',
        'batch_size' : 1,
    }
    tsds = Shp(config=test_config)
    tsdl = DataLoader(tsds, batch_size=test_config['batch_size'], shuffle=False)
    print('got val dl', len(vdl))

    if train_config['use_model'] == 1:
        mdl = Mdl()
    elif train_config['use_model'] == 2:
        mdl = Mdl_m()
    elif train_config['use_model'] == 3:
        mdl = Mdl_s()
    elif train_config['use_model'] == 4:
        mdl = Mdl_c()
    print('model initialized')
    print(mdl)
    total_params = 0
    for params in mdl.parameters():
        num_params = 1
        for x in params.size():
            num_params *= x
        total_params += num_params
    print('Totoal model params = ', total_params)
    mdl = mdl.cuda()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, mdl.parameters()), lr=train_config['lr'])
    print('Optim initialized')
    print(opt)
    
    l = Loss()
    print('Loss initialized')
    print(l)

    wt = SummaryWriter(log_dir= osp.join(train_config['root'],'runs', train_config['cache_file']))
    for t in train_config :
        print(t, ': \t', train_config[t])
    for t in val_config :
        print(t, ': \t', val_config[t])
    for t in test_config :
        print(t, ': \t', test_config[t])

    for ep in range(train_config['epoch']) :
        tr_iter = iter(tdl)
        print('got train iter', len(tr_iter))
        running_loss = []
        running_loss_v = []
        running_accuracy = []
        running_accuracy_v = []
        mdl.train()
        for ix, data in enumerate(tr_iter) :
            gc.collect()
            image, gt_lbl, ann, pc = data
            image, gt_lbl = image.cuda(), gt_lbl.cuda()
            opt.zero_grad()        
            hm, emb = mdl(image)
            tr_l = l(hm, emb, gt_lbl, ann, pc, 'train', wt, ep, ix)            
            running_loss.append(tr_l.item())
            mae = get_mae(hm, gt_lbl)
            # mae, prec = get_mae(hm, gt_lbl)
            print('MAE ', sum(mae)/len(mae))            
            # print('prec ', prec)            
            for _ in mae :
                running_accuracy.append(_)
            tr_l.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), max_norm=1)
            opt.step()
            # break
        print('Train Ep', ep, 'Loss :', sum(running_loss)/len(running_loss), 'Acc', 1 - sum(running_accuracy)/len(running_accuracy))        
        if ep% val_config['freq'] == 0 :
            v_iter = iter(vdl)
            print('got val iter', len(v_iter))
            mdl.eval()
            with torch.no_grad():
                for ix, data in enumerate(v_iter) :
                    gc.collect()
                    image, gt_lbl, ann, pc = data
                    image, gt_lbl = image.cuda(), gt_lbl.cuda()
                    opt.zero_grad()        
                    hm, emb = mdl(image)            
                    tr_l = l(hm, emb, gt_lbl, ann, pc, 'val', wt, ep, ix)            
                    running_loss_v.append(tr_l.item())
                    # mae, prec = get_mae(hm, gt_lbl)
                    mae = get_mae(hm, gt_lbl)
                    print('MAE ', sum(mae)/len(mae))            
                    # print('prec ', prec)            
                    for _ in mae :
                        running_accuracy_v.append(_)
                    # break
            print('Valid Ep', ep, 'Loss :', sum(running_loss_v)/len(running_loss_v), 'Acc', 1- sum(running_accuracy_v)/len(running_accuracy_v))        

        params = {'mod_wt_' : mdl.state_dict(), 'optim_wt_' : opt.state_dict() }
        sv = osp.join(train_config['root'],'saves', train_config['cache_file'])
        if not osp.isdir(sv) :
            os.mkdir(sv)
        sv = osp.join(sv,  '__'+str(ep))
        with open(sv, "wb") as f:
            torch.save(params, f)