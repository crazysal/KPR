import torch
import torch.nn as nn
from torch.nn import functional as F 
from torch.utils.tensorboard import SummaryWriter
import pickle 
import copy
import numpy as np
import json
import os
import os.path as osp
import PIL 
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from tqdm import tqdm 


from collections import deque
import matplotlib 
from matplotlib import pyplot as plt

from hg_net import hg
from chart_transform import ChartTransform

class Point_Detector_Lenovo_FCN(nn.Module):
    def __init__(self, **kwargs) :
        super(Point_Detector_Lenovo_FCN, self).__init__()
        n_classes = kwargs['n_classes']
        self.loss = kwargs['loss']
        self.intit = nn.Conv2d(1, 64, 7, padding=3 ) ## N, 1, 128, 128, -> N, 64, 128, 128

        self.stage1 = nn.Conv2d(64, 128, 5, padding=2 ) ## N, 64, 128, 128, -> N, 128, 128, 128
        self.stage1BN1 = torch.nn.BatchNorm2d(128)
        self.stage1p = nn.MaxPool2d(2) ## N, 128, 128, 128, -> N, 128, 64, 64

        self.stage1_ = nn.Conv2d(128, 128, 3, padding=1 ) ## N, 128, 64, 64, -> N, 128, 64, 64
        self.stage1BN2 = torch.nn.BatchNorm2d(128)
        self.stage1_p = nn.MaxPool2d(2) ## N, 128, 64, 64, -> N, 128, 32, 32

        self.stage2 = nn.Conv2d(128, 256, 3, padding=1 ) ## N, 128, 32, 128, -> N, 256, 32, 32
        self.stage2BN1 = torch.nn.BatchNorm2d(256)
        self.stage2p = nn.MaxPool2d(2) ## N, 256, 32, 32, -> N, 256, 16, 16

        self.stage2_ = nn.Conv2d(256, 256, 3, padding=1 ) ## N, 256, 16, 16, -> N, 256, 16, 16
        self.stage2BN2 = torch.nn.BatchNorm2d(256)
        self.stage2_p = nn.MaxPool2d(2) ## N, 256, 16, 16, -> N, 256, 8, 8


        self.up_stage2_ = nn.ConvTranspose2d(256, 256, 3, dilation=4) ## N, 256, 8, 8 -> N, 256, 16, 16

        self.up_stage2 = nn.ConvTranspose2d(256, 128, 3, dilation=8) ## N, 256, 16, 16 -> N, 128, 32, 32

        self.up_stage1_ = nn.ConvTranspose2d(128, 128, 3, dilation=16) ## N, 128, 32, 32 -> N, 128, 64, 64

        self.up_stage1 = nn.ConvTranspose2d(128, 64, 3, dilation=32) ## N, 128, 16, 16 -> N, 64, 32, 32

        self.hmap = nn.Conv2d(64, n_classes, 1)
        if self.loss == 'all' :
            self.hmap2 = nn.Conv2d(64, n_classes+1, 1)
        # self._initialize_weights()

    def forward(self, a) :
        a_ = self.intit(a)
        a_4 = self.stage1p(self.stage1BN1(F.leaky_relu_(self.stage1(a_))))
        a_3 = self.stage1_p(self.stage1BN2(F.leaky_relu_(self.stage1_(a_4))))
        a_2 = self.stage2p(self.stage2BN1(F.leaky_relu_(self.stage2(a_3))))
        a_1 = self.stage2_p(self.stage2BN2(F.leaky_relu_(self.stage2_(a_2))))
       
        # a_ = self.intit(a)
        # a_4 = self.stage1p(F.leaky_relu_(self.stage1(a_)))
        # a_3 = self.stage1_p(F.leaky_relu_(self.stage1_(a_4)))
        # a_2 = self.stage2p(F.leaky_relu_(self.stage2(a_3)))
        # a_1 = self.stage2_p(F.leaky_relu_(self.stage2_(a_2)))

        au_1 = self.up_stage2_(F.leaky_relu_(a_1))
        au_1 += a_2
        au_2 = self.up_stage2(F.leaky_relu_(au_1))
        au_2 += a_3
        au_3 = self.up_stage1_(F.leaky_relu_(au_2))
        au_3 += a_4
        au_4 = self.up_stage1(F.leaky_relu_(au_3))
        if self.loss == 'softmax' :
            op = F.softmax(self.hmap(au_4),  dim=1)
        # print('op.shape', op.shape)
        # print('torch.sum(op, dim = 0)', torch.sum(op, dim = 0))
        # print('torch.sum(op, dim = 1)', torch.sum(op, dim = 1))
        # print('torch.sum(op, dim = 2)', torch.sum(op, dim = 2))
        # print('torch.sum(op, dim = 3)', torch.sum(op, dim = 3))
        elif self.loss == 'BCE' or self.loss == 'MSE':
            op = torch.sigmoid(self.hmap(au_4))
        elif self.loss == 'all' :
            op = (torch.sigmoid(self.hmap(au_4)), F.softmax(self.hmap2(au_4),  dim=1))
        return op
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             m.weight.data.zero_()
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         if isinstance(m, nn.ConvTranspose2d):
    #             assert m.kernel_size[0] == m.kernel_size[1]
    #             initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
    #             m.weight.data.copy_(initial_weight)


        

'''
    intit = nn.Conv2d(1, 64, 7, padding=3 ) ## N, 1, 128, 128, -> N, 64, 128, 128

    stage1 = nn.Conv2d(64, 128, 5, padding=2 ) ## N, 64, 128, 128, -> N, 128, 128, 128
    stage1p = nn.MaxPool2d(2) ## N, 128, 128, 128, -> N, 128, 64, 64

    stage1_ = nn.Conv2d(128, 128, 3, padding=1 ) ## N, 128, 64, 64, -> N, 128, 64, 64
    stage1_p = nn.MaxPool2d(2) ## N, 128, 64, 64, -> N, 128, 32, 32

    stage2 = nn.Conv2d(128, 256, 3, padding=1 ) ## N, 128, 32, 128, -> N, 256, 32, 32
    stage2p = nn.MaxPool2d(2) ## N, 256, 32, 32, -> N, 256, 16, 16

    stage2_ = nn.Conv2d(256, 256, 3, padding=1 ) ## N, 256, 16, 16, -> N, 256, 16, 16
    stage2_p = nn.MaxPool2d(2) ## N, 256, 16, 16, -> N, 256, 8, 8


    up_stage2_ = nn.ConvTranspose2d(256, 256, 3, dilation=4) ## N, 256, 8, 8 -> N, 256, 16, 16

    up_stage2 = nn.ConvTranspose2d(256, 128, 3, dilation=8) ## N, 256, 16, 16 -> N, 128, 32, 32

    up_stage1_ = nn.ConvTranspose2d(128, 128, 3, dilation=16) ## N, 128, 32, 32 -> N, 128, 64, 64

    up_stage1 = nn.ConvTranspose2d(128, 64, 3, dilation=32) ## N, 128, 16, 16 -> N, 64, 32, 32

    hmap = nn.Conv2d(64, 3, 1)

    a_ = intit(a)
    a_4 = stage1p(F.leaky_relu_(stage1(a_)))
    a_3 = stage1_p(F.leaky_relu_(stage1_(a_4)))
    a_2 = stage2p(F.leaky_relu_(stage2(a_3)))
    a_1 = stage2_p(F.leaky_relu_(stage2_(a_2)))

    au_1 = up_stage2_(F.leaky_relu_(a_1))
    au_1 += a_2
    au_2 = up_stage2(F.leaky_relu_(au_1))
    au_2 += a_3
    au_3 = up_stage1_(F.leaky_relu_(au_2))
    au_3 += a_4
    au_4 = up_stage1(F.leaky_relu_(au_3))

    a.shape
    a_.shape
    a_4.shape
    a_3.shape
    a_2.shape
    a_1.shape
    au_1.shape
    au_2.shape
    au_3.shape
    au_4.shape
'''




class PMC_line(Dataset):
    def __init__(self, split='train', loss='BCE'):
        self.split = split
        self.lss = loss
        self.im_dir = '/data/ChartsInfo/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/line/'
        self.im_dir = '/home/sahmed9/reps/DeepRule/utils/data/images/'
        self.json_dir = '/home/sahmed9/reps/DeepRule/utils/data/JSONs/'
        #line
        # self.mask_dir = '/home/sahmed9/reps/DeepRule/utils/data/line_mask2/'
        # fl_list = '/home/sahmed9/reps/DeepRule/utils/data/split4_train_task6_line.pkl'
        #allPMC
        self.mask_dir = '/home/sahmed9/reps/DeepRule/utils/data/all_mask/'
        # self.mask_dir = '/home/sahmed9/reps/DeepRule/utils/data/all_mask_32/'
        fl_list1 = '/home/sahmed9/reps/DeepRule/utils/data/split4_train_task6_all.pkl'
        #allPMC U Synth
        fl_list2 = '/home/sahmed9/reps/DeepRule/utils/data/train_task6_syntyPMC.pkl'

        self.im_test_dir = '/home/sahmed9/reps/DeepRule/utils/data/images_line/'
        self.json_test_dir = '/home/sahmed9/reps/DeepRule/utils/data/line_json_test/'
        self.mean_bgr = np.array([244.4647716277776, 220.6868814349762, 220.9919510619279])
        self.std_bgr = np.array([39.9591211035063, 37.645456502289086, 36.70677014435])

        chart_types=np.array(['area',
            'heatmap',
            'horizontal_bar',
            'hrorizontal_interval',
            'line',
            'manhattan',
            'map',
            'pie',
            'scatter',
            'scatter-line',
            'surface',
            'venn',
            'vertical_bar',
            'vertical_box',
            'vertical_interval'])
        TRAIN_ALL = True
        self.crop = False
        self.ct = ChartTransform()
        ff1 = open(fl_list1, 'rb')
        ff1 = pickle.load(ff1)
        ff2 = open(fl_list2, 'rb')
        ff2 = pickle.load(ff2)
        if  TRAIN_ALL :
            ff= [osp.join(folder, file) for folder in ff1 for file in ff1[folder] ]  + [osp.join(folder, file) for folder in ff2 for file in ff2[folder] ]        
        splt_val = int(0.85 * len(ff))
        self.files={
            'train' : ff[:splt_val],
            'val' : ff[splt_val:],
            'test': os.listdir(self.json_test_dir)
            }
        print('Loaded Total {} charts {} train, {} val from im dir {} and json dir {}'.format(len(ff), len(self.files['train']),len(self.files['val']), self.im_dir, self.json_dir))

    def __len__(self):
        return len(self.files[self.split])
    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        if self.split == 'test': 
            img_file = osp.join(self.im_test_dir, data_file[:-4]+'jpg')
            if not osp.isfile(img_file) :
                img_file = osp.join(self.im_test_dir, data_file[:-4]+'png')
            lbl_file = osp.join(self.json_test_dir, data_file)
        else : 
            img_file = osp.join(self.im_dir,data_file[:-4]+'jpg')
            if not osp.isfile(img_file) :
                img_file = osp.join(self.im_dir,data_file[:-4]+'png')
            lbl_file = osp.join(self.mask_dir, data_file[:-4]+'npy')
        
        # load image
        img = PIL.Image.open(img_file)
        w, h = img.size
        if self.split == 'test' :
            lb = json.load(open(lbl_file,'r'))
        else : 
            lb = json.load(open(osp.join(self.json_dir, data_file), 'r'))
        ploth, plotw , x0, y0 = lb['task6']['input']['task4_output']['_plot_bb'].values()
        # print('og ', img.size)
        if self.crop :
            img = img.crop((x0, y0, x0+plotw, y0+ploth))
        # print('after crop ', img.size)        
        img = img.convert('RGB')
        # print('after RGB ', img.size)        
        img = self.ct(chart=img,js_obj=lb)
        # print('after process ', img.shape)
        img = img.float()
        # print('*'*80)
        img.requires_grad = True
        # load label
        if self.split == 'test': 
            lbl = lb
            text_val = lbl['task6']['input']['task2_output']['text_blocks']
            # print('text_val', text_val)
            x_axs = lbl['task6']['input']['task4_output']['axes']['x-axis']
            # print('x_axs', x_axs)
            for pt in x_axs :
                id_ = pt['id']
                for block in text_val :
                    if block['id'] == id_ :
                        pt['text'] = block['text']
            y_axs = lbl['task6']['input']['task4_output']['axes']['y-axis']
            for pt in y_axs :
                id_ = pt['id']
                for block in text_val :
                    if block['id'] == id_ :
                        pt['text'] = block['text']            
            gt = [lbl['task6']['input']['task4_output']['_plot_bb'], x_axs, y_axs, (w,h), data_file[:-5] ]
        else : 
            # print('lbl_file',lbl_file)
            lbl = np.load(lbl_file)
            # print('lbl1_',lbl)
            # print('lbl1',lbl.shape)

            lbl = lbl.astype(np.float)

            lbl1 = copy.deepcopy(lbl)
            lbl = gaussian_filter(lbl, 0.3)
            lbl1 = gaussian_filter(lbl1, 1)
            lbl = torch.from_numpy(lbl)
            lbl1 = torch.from_numpy(lbl1).unsqueeze(0)
            # lbl = torch.from_numpy(lbl)
            # print('lbl2',lbl.shape, torch.sum(lbl))
            return img.float(), (lbl, lbl1.float())
            # return img.float(), lbl.long()
        # print('img , lbl == ', img.shape, lbl.shape)
        # assert(img.shape == lbl.shape)
        # assert(img.shape == lbl.shape)
        # print('img , lbl after== ', img.shape, lbl.shape)
        # print('img , lbl after== ', img, lbl)
        return img.float(), gt


class WeightedMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(WeightedMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = torch.sigmoid(heatmaps_pred[idx].squeeze())
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                # loss += 0.5 * self.criterion(
                loss += self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

"""
class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints
"""

def run_train():
    arg = {
        'Batch_size' : 20,
        'Epochs' : 20,
        'lr' : 0.00025,
        'cache_dir' : '/home/sahmed9/reps/DeepRule/utils/cache/',
        'cache_file' : 'AllPMC_PointDetectorHgNet_8b', 
        'n_class' : 2, 
        'loss' : 'wtMSE-bce-ce', 
        'model' : 'hg_3_1',
        'cuda':True
    }
    print(arg)
    train_ds = PMC_line('train', arg['loss'])
    train_dataloader = DataLoader(train_ds, batch_size=arg['Batch_size'], shuffle=True)

    val_ds = PMC_line('val', arg['loss'])
    val_dataloader = DataLoader(val_ds, batch_size=arg['Batch_size'], shuffle=True)

    if arg['model'] == 'hg_3_1' :
        model  = hg(num_stacks=3, num_blocks=1, num_classes=arg['n_class'])
    else :
        model = Point_Detector_Lenovo_FCN(n_classes=arg['n_class'], loss=arg['loss'])
    
    writer = SummaryWriter(log_dir=osp.join(arg['cache_dir'],'runs', arg['cache_file']))
    # writer.add_graph(model, input_to_model=torch.randn(1, 3, 512, 512))
    # exit()
    
    if arg['cuda'] :
        model = model.cuda()
    print(model)
    total_params = 0
    for params in model.parameters():
        num_params = 1
        for x in params.size():
            num_params *= x
        total_params += num_params
    print("total parameters: {}".format(total_params))

    
    if arg['loss'] == 'BCE' :
        loss = nn.BCELoss()
    elif arg['loss'] == 'softmax': 
        loss = nn.CrossEntropyLoss()
    elif arg['loss'] == 'MSE': 
        loss = nn.MSELoss()
    elif arg['loss'] == 'wtMSE': 
        loss = WeightedMSELoss()
        loss = nn.MSELoss()
    elif arg['loss'] == 'wtMSE-bce': 
        loss =  (WeightedMSELoss(), F.binary_cross_entropy)
    elif arg['loss'] == 'wtMSE-bce-ce': 
        loss = (WeightedMSELoss(), F.binary_cross_entropy_with_logits , nn.CrossEntropyLoss(weight=torch.tensor([0.001, 0.99]).cuda()))
    elif arg['loss'] == 'all': 
        # loss = (nn.MSELoss(), nn.BCELoss, nn.CrossEntropyLoss(weight=torch.tensor([0.05, 0.95])))
        loss = (nn.MSELoss(), F.binary_cross_entropy , nn.CrossEntropyLoss(weight=torch.tensor([0.001, 0.99]).cuda()))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    

    lr = arg['lr']
    print('Loss fn, {} LR : {}, Optimizer : {}'.format(loss, lr, optimizer))
    for ep in tqdm(range(arg['Epochs'])) :
    # for ep in range(arg['Epochs']) :
        # '''
        model.train()
        print('Traainig Epoch', ep)
        tr_iter = iter(train_dataloader)
        for ix, data in enumerate(tr_iter) :
            im, lbls = data
            lbl_ce, lbl_bce = lbls
            
            # print('*'*80)
            # print('im', im.shape)
            # print('im mimax', torch.min(im), torch.max(im))
            # print('lbl', lbl.shape)
            # print('lb', lb.shape)
            # print('lbl mimax', torch.min(lbl), torch.max(lbl))
            if arg['cuda'] :
                im = im.cuda()
                lbl_bce = lbl_bce.cuda()
                lbl_ce = lbl_ce.cuda()
            lbl_mse = [lbl_bce, 1-lbl_bce]
            lbl_mse = torch.cat(lbl_mse, dim=1) 
            hm  = model(im)
            # print(len(hm))
            # for h in hm : 
                # print('hm ::::', h.shape)
                # print('hm mimax', torch.min(h), torch.max(h))
            # exit()
            wt_bce = torch.ones_like (lbl_bce) / 99.5  +  (1.0 - 1.0 / 99.5) * lbl_bce
            wt_mse = torch.tensor([[0.99, 0.01]])
            if arg['cuda'] :
                wt_mse = wt_mse.cuda()
            optimizer.zero_grad()
            if arg['loss'] == 'all' :
                lbl2 = lbl.squeeze(dim=1).long()
                # np.unravel_index(np.argsort(canvas.ravel()), canvas.shape)
                lbl2[lbl2<0.33] = 0
                lbl2[lbl2>0] = 1 
                l1= loss[0](hm[0], lbl) 
                l2= loss[1](hm[0], lbl, weight=wt)
                l3= loss[2](hm[1],lbl2) 
                training_loss = 0.1*l1+0.7*l2+0.2*l3
            elif arg['loss'] == 'wtMSE' :
                loss_ = 0
                for o in hm:
                    loss_ += loss(o, lbl, target_weight)
                    # loss_ += loss(o, lbl)
                training_loss = loss_
            elif arg['loss'] == 'wtMSE-bce' :
                # print('lbl->',  lbl.shape, 'inp ->', (hm[0][:, 0, :, :].unsqueeze(dim=1)).shape)
                fm1 = torch.sigmoid(hm[0][:, 0, :, :].unsqueeze(dim=1))
                fm2 = torch.sigmoid(hm[0][:, 1, :, :].unsqueeze(dim=1))
                l2 = (loss[1](fm1, lbl, weight=wt) + loss[1](fm2, lbl, weight=wt)) /2
                
                # print(hm[1].shape, lb.shape)
                l1 = loss[0](hm[1], lb, target_weight) 
                # loss_ += loss(o, lbl)
                training_loss = l1+l2
                print(ix, "Train Loss =", round(l1.item(),4), round(l2.item(),4), round(training_loss.item(), 4))            
            elif arg['loss'] == 'wtMSE-bce-ce' :
                ## MSE 
                l1 = loss[0](hm[0], lbl_mse, wt_mse) 

                ### BCE
                # print('lbl->',  lbl.shape, 'inp ->', (hm[0][:, 0, :, :].unsqueeze(dim=1)).shape)
                fm1 = hm[1][:, 0, :, :].unsqueeze(dim=1)
                fm2 = hm[1][:, 1, :, :].unsqueeze(dim=1)
                l2 = (loss[1](fm1, lbl_bce, weight=wt_bce) + loss[1](fm2, lbl_bce, weight=wt_bce)) /2
                
                # print(hm[1].shape, lb.shape)
                ### CE
                fm = torch.softmax(hm[2],dim=1)
                lbl_ce = lbl_ce.squeeze(dim=1).long()
                lbl_ce[lbl_ce<0.33] = 0
                lbl_ce[lbl_ce>0] = 1 
                l3 = loss[2](fm, lbl_ce) 

                # loss_ += loss(o, lbl)
                training_loss = 0.3*l1+0.5*l2 +0.2*l3
            else :
                    training_loss = loss(hm, lbl)

            training_loss.backward()
            optimizer.step()
            # print(ix, "Train Loss", training_loss.item())
            # exit()
            if arg['loss'] == 'all' or arg['loss'] == 'wtMSE-bce-ce':
                print(ix, "Train Loss =", l1.item(), l2.item(), l3.item(), training_loss.item()) 
                writer.add_scalar("Train Loss MSE", l1.item(), (ep+1)*(ix+1))
                writer.add_scalar("Train Loss BCE", l2.item(), (ep+1)*(ix+1))
                writer.add_scalar("Train Loss CE", l3.item(), (ep+1)*(ix+1))
            writer.add_scalar("Train Loss", training_loss.item(), (ep+1)*(ix+1))

        # exit()
        # '''
        tr_iter = iter(val_dataloader)
        model.eval()
        with torch.no_grad() :
            print('Eval Epoch', ep)
            for ix, data in enumerate(tr_iter) :
                im, lbl = data
                im = im.cuda()
                lbl_ce, lbl_bce = lbl
                # lbl_ = lbl.cuda()
                # print(im.shape)
                # print(lbl.shape)
                hm  = model(im)
                wt_bce = torch.ones_like (lbl_bce) / 99.5  +  (1.0 - 1.0 / 99.5) * lbl_bce
                wt_mse = torch.tensor([[0.99, 0.01]])
                if arg['cuda'] :
                    lbl_bce = lbl_bce.cuda()
                    lbl_ce = lbl_ce.cuda().long()
                    wt_bce = wt_bce.cuda()
                    wt_mse = wt_mse.cuda()
                
                # print(hm[0].shape, hm[1].shape)
                if arg['loss'] == 'all' :
                    lbl2 = lbl.squeeze(dim=1).long()
                    lbl2[lbl2<0.33] = 0
                    lbl2[lbl2>0] = 1 
                    l1= loss[0](hm[0], lbl) 
                    l2= loss[1](hm[0], lbl, weight=wt)
                    l3= loss[2](hm[1],lbl2) 
                    val_loss =  0.1*l1+0.7*l2+0.2*l3
                elif arg['loss'] == 'wtMSE-bce' :
                    lb = [lbl, 1-lbl]
                    lb = torch.cat(lb, dim=1) 
                    # print('lbl->',  lbl.shape, 'inp ->', (hm[0][:, 0, :, :].unsqueeze(dim=1)).shape)
                    fm1 = torch.sigmoid(hm[0][:, 0, :, :].unsqueeze(dim=1))
                    fm2 = torch.sigmoid(hm[0][:, 1, :, :].unsqueeze(dim=1))
                    l2= (loss[1](fm1, lbl, weight=wt) + loss[1](fm2, lbl, weight=wt)) /2
                    
                    # print(hm[1].shape, lb.shape)
                    l1= loss[0](hm[1], lb, target_weight) 
                    # loss_ += loss(o, lbl)
                    val_loss = l1+l2
                elif arg['loss'] == 'wtMSE-bce-ce' :
                    # print('lbl->',  lbl.shape, 'inp ->', (hm[0][:, 0, :, :].unsqueeze(dim=1)).shape)
                    fm1 = hm[0][:, 0, :, :].unsqueeze(dim=1)
                    fm2 = hm[0][:, 1, :, :].unsqueeze(dim=1)
                    l2 = (loss[1](fm1, lbl_bce, weight=wt_bce) + loss[1](fm2, lbl_bce, weight=wt_bce)) /2
                    
                    lbl_mse = [lbl_bce, 1-lbl_bce]
                    lbl_mse = torch.cat(lbl_mse, dim=1) 
                    # print(hm[1].shape, lb1.shape)
                    l1= loss[0](hm[1], lbl_mse, wt_mse) 
                    
                    
                    fm = torch.softmax(hm[2], dim=1)
                    lbl_ce[lbl_ce<0.33] = 0
                    lbl_ce[lbl_ce>0] = 1 
                    l3 = loss[2](fm, lbl_ce) 

                    # loss_ += loss(o, lbl)
                    # val_loss = 0.1*l1+0.7*l2 +0.2*l3
                    val_loss = 0.3*l1+0.5*l2 +0.2*l3
                                
                else :
                    val_loss = loss(hm, lbl)
                print("Val Loss", val_loss.item())
                # exit()
                writer.add_scalar("Val Loss", val_loss.item(), (ep+1)*(ix+1))
        
        if ep % 4 == 0 : 
            lr/=10
            print('Epoch {} LR Updated from {} to {}'.format(ep, lr*10, lr))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        
        sv = osp.join(arg['cache_dir'],'saves',  '_ep_'+str(ep)+'_model_'+arg['cache_file']+'.pth')
        with open(sv, "wb") as f:
            params = {
                'mod_wt_' : model.state_dict(), 
                'optim_wt_' : optimizer.state_dict()
                }
            # torch.save(params, f, _use_new_zipfile_serialization=False)
            torch.save(params, f)



    


'''
    >>> len(x_sum)
    741
    >>> sum(x_sum)
    181148.3957761832
    >>> sum(x_sum)/len(x_sum)
    244.4647716277776

    len(y_sum)
    741
    >>> sum(y_sum)/len(y_sum)
    220.6868814349762

    >>> sum(z_sum)/len(z_sum)
    220.9919510619279

    >>> sum(x_std)/len(x_std)
    39.9591211035063

    >>> sum(y_std)/len(y_std)
    37.645456502289086

    >>> sum(z_std)/len(z_std)
    36.70677014435

    T.Normalize(mean=(244.4647716277776, 220.6868814349762, 220.9919510619279), std=(39.9591211035063, 37.645456502289086, 36.70677014435 ))
    pre_process = T.Compose([T.Grayscale(num_output_channels=1), T.ToTensor(),T.Normalize(mean=0.5, std=0.5)])

'''
# x= {'line/PMC2674413___1297-9686-41-32-6.json' :1, 'line/PMC4568336___CMMM2015-515613.008.json' :1, 'line/PMC2898822___1471-2458-10-299-6.json': 4, 'line/PMC4568336___CMMM2015-515613.006.json':8}
## Run locally on chort
# xcept = {'vbox/59253.json':'weird crop','vbox/132011.json':'weird crop','vbox/41388.json':'weird crop','vbox/46635.json':'weird crop'}
def geerate_masks(choice = 2, map_sz=None, xpt = None) :
    if choice == 1 :
        #PMC Line only
        img_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/line'
        json_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/line'
        fl_list = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/split4_train_task6_line.pkl'
    elif choice == 2 :
        # Synth 
        img_dir = '/home/sahmed9/Documents/data/charts/ICPR_ChartCompetition2020_AdobeData/Chart-Images-and-Metadata/ICPR/Charts/'
        json_dir = '/home/sahmed9/Documents/data/charts/ICPR_ChartCompetition2020_AdobeData/Task-level-JSONs/JSONs/'
        fl_list = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/train_task6_syntyPMC.pkl'
        sd = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/synth_5_mask/'
        sd = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/synth_5_mask_plotbb_128/'
        sd = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/synth_5_mask_interpolate_128/'
    elif choice == 3 :
        #PMC ALL
        img_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/'
        json_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON'
        fl_list = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/split4_train_task6_all.pkl'
        sd = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/all_pmc_mask/'
        sd = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/all_pmc_mask_plotbb_128/'
        sd = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/all_pmc_mask_interpolate_128/'
    os.mkdir(sd) if not osp.isdir(sd) else print(sd, 'created')
    if xpt is None :
        # xcept= {'line/PMC2674413___1297-9686-41-32-6.json' :1, 'line/PMC4568336___CMMM2015-515613.008.json' :1, 'line/PMC2898822___1471-2458-10-299-6.json': 4}
        # xcept = {'vbox/59253.json':'weird crop'}
        xcept = []
    else :
        xcept = xpt
    ff = pickle.load(open(fl_list, 'rb'))
    # ff = [osp.join(folder, file) for folder in ff for file in ff[folder] ]
    # ff = [osp.join(folder, file) for folder in ff for file in ff[folder] if folder in ['vertical_box', 'vertical_bar', 'horizontal_bar','hbox','hGroup','hStack','line','scatter','line2','scatter2', 'vbox','vGroup','vStack']]
    ff = [osp.join(folder, file) for folder in ff for file in ff[folder] if folder in ['polar', 'pie', 'donut','area']]
    # ff = [osp.join(folder, file) for folder in ff for file in ff[folder] if folder in ['hbox','hGroup','hStack','line','scatter','vbox','vGroup','vStack']]
    # ff = [osp.join(folder, file) for folder in ff for file in ff[folder] if folder in ['hGroup','hStack','vGroup','vStack']]
    # ff = [osp.join(folder, file) for folder in ff for file in ff[folder] if folder in ['line2','scatter2','vertical_box']]
    # ff = [osp.join(folder, file) for folder in ff for file in ff[folder] if folder in ['scatter2']]
    if map_sz is None :
        map_sz = 128
    # # map_sz = 32
    # x_sum = []
    # y_sum = []
    # z_sum = []
    # x_std = []
    # y_std = []
    # z_std = []
    for ix, f in enumerate(ff) : 
        if f in xcept :
            print('\n :::', ix,'::: Skipping for ', f)
        else :
            print('\n :::', ix,'::: running for ', f)
            if choice == 1 or choice == 3 :
                img_file = osp.join(img_dir, f[:-4]+'jpg')
            else : 
                img_file = osp.join(img_dir, f[:-4]+'png')
            img = PIL.Image.open(img_file)
            # img.show()
            # trf = T.Grayscale(num_output_channels=1)
            # ary = np.array(trf(img), dtype=np.uint8)
            # sp = ary.shape
            # px = sp[0]*sp[1]
            # x_sum.append( (ary[:, :].sum())/px if len(sp) == 2 else (ary[:, :, 0].sum())/px )
            # y_sum.append(0 if len(sp) == 2 else (ary[:, :, 1].sum()  )/px )
            # z_sum.append(0  if len(sp) == 2 else (ary[:, :, 2].sum())/px )        
            # x_std.append( np.std(ary[:, :]) if len(sp) == 2 else np.std(ary[:, :, 0]) )
            # y_std.append(0 if len(sp) == 2 else np.std(ary[:, :, 1]))
            # z_std.append(0  if len(sp) == 2 else np.std(ary[:, :, 2]))
            # canvas = torch.zeros(img.size)
            canvas = torch.zeros((map_sz,map_sz))
            width, height = img.size
            lbl_file = osp.join(json_dir, f)
            lbl_file = open(lbl_file, 'r')
            js_obj = json.load(lbl_file)
            print('wh', width, height)
            # height, width, x0, y0 = js_obj['task6']['input']['task4_output']['_plot_bb'].values()
            # print('plothwh', width, height)
            chart_type = f.split('/')[0]
            print('chart_type', chart_type)
            if chart_type == 'line' or chart_type == 'line2' :
                ln_data = js_obj['task6']['output']['visual elements']['lines']
                for l in ln_data : 
                    for pt in l : 
                        # xs = int((pt['x']- x0)*map_sz/width)
                        # ys = int((pt['y']- y0)*map_sz/height)                    
                        xs = int((pt['x'])*map_sz/width)
                        ys = int((pt['y'])*map_sz/height)                    
                        if xs == map_sz :
                            xs-=1
                        if ys == map_sz :
                            ys-=1
                        if xs < map_sz and ys < map_sz :
                            print(pt['x'], pt['y'], '->', xs, ys)
                            canvas[ys,xs] = 1
                        else: 
                            print('ERROR', xs, ys, map_sz, f)
                            print(lod)
            elif chart_type == 'vertical_box' or chart_type == 'vbox' or chart_type == 'hbox':
                box_data = js_obj['task6']['output']['visual elements']['boxplots']
                for box in box_data : 
                    for pt in  box: 
                        # xs = int((box[pt]['x']- x0)*map_sz/width)
                        xs = int((box[pt]['x'])*map_sz/width)
                        # ys = int((box[pt]['y']- y0)*map_sz/height)
                        ys = int((box[pt]['y'])*map_sz/height)
                        print(pt,box[pt]['x'], box[pt]['y'] ,'->', xs, ys )
                        if xs == map_sz :
                            xs-=1
                        if ys == map_sz :
                            ys-=1
                        if xs < map_sz and ys < map_sz :
                            canvas[ys,xs] = 1
                        # else: 
                            # print(pt,box[pt]['x'], box[pt]['y'] ,'->' , xs, '/',width , ys, '/', height, map_sz,  f)
                            # print(lod)
            elif chart_type == 'scatter2':
                pt_data = js_obj['task6']['output']['visual elements']['scatter points']
                print(len(pt_data))
                for pt in  pt_data : 
                    # xs = int((pt['x']- x0)*map_sz/width)
                    # ys = int((pt['y']- y0)*map_sz/height)
                    xs = int((pt['x'])*map_sz/width)
                    ys = int((pt['y'])*map_sz/height)
                    if xs == map_sz :
                        xs-=1
                    if ys == map_sz :
                        ys-=1
                    if xs < map_sz and ys < map_sz :
                        canvas[ys,xs] = 1
                    else: 
                        print('ERROR', xs, ys, map_sz, f)
                        print(lod)
            elif chart_type == 'scatter':
                pt_data = js_obj['task6']['output']['visual elements']['scatter points']
                print(len(pt_data))
                for pts in pt_data : 
                    for pt in  pts: 
                        # xs = int((pt['x']- x0)*map_sz/width)
                        xs = int((pt['x'])*map_sz/width)
                        ys = int((pt['y'])*map_sz/height)
                        # ys = int((pt['y']- y0)*map_sz/height)
                        if xs == map_sz :
                            xs-=1
                        if ys == map_sz :
                            ys-=1
                        if xs < map_sz and ys < map_sz :
                            canvas[ys,xs] = 1
                        else: 
                            print('ERROR', xs, ys, map_sz, f)   
                            print(lod)
            elif chart_type == 'horizontal_bar' or chart_type =='hGroup'  or chart_type =='hStack':
                bar_data = js_obj['task6']['output']['visual elements']['bars']
                for bars in bar_data : 
                    ## bottom left
                    # xs = int((bars['x0']- x0)*map_sz/width)
                    # ys = int((bars['y0']- y0)*map_sz/height)
                    xs = int((bars['x0'])*map_sz/width)
                    ys = int((bars['y0'])*map_sz/height)
                    print(bars['x0'], '->', xs,bars['y0'], '->', ys )  
                    if xs == map_sz :
                        xs-=1
                    if ys == map_sz :
                        ys-=1
                    if xs < map_sz and ys < map_sz :
                        canvas[ys,xs] = 1
                    else: 
                        print('ERROR', xs, ys, map_sz, f)
                        print(lod)
                    ## top right
                    # xs = int(((bars['x0']- x0)+bars['width'])*map_sz/width)          
                    # ys = int(((bars['y0']- y0)+bars['height'])*map_sz/height)    
                    xs = int(((bars['x0'])+bars['width'])*map_sz/width)          
                    ys = int(((bars['y0'])+bars['height'])*map_sz/height)    
                    print(bars['x0'] + bars['width'], '->', xs, bars['y0']+bars['height'], '->', ys )  
                    if xs == map_sz :
                        xs-=1
                    if ys == map_sz :
                        ys-=1
                    if xs < map_sz and ys < map_sz :
                        canvas[ys,xs] = 1
                    else: 
                        print('ERROR', xs, ys, map_sz, f)
                        print(lod)
            elif chart_type == 'vertical_bar'  or chart_type =='vStack'  or chart_type =='vGroup':
                bar_data = js_obj['task6']['output']['visual elements']['bars']
                for bars in bar_data : 
                    ## bottom left
                    # xs = int((bars['x0']- x0)*map_sz/width)
                    # ys = int((bars['y0']- y0)*map_sz/height)
                    xs = int((bars['x0'])*map_sz/width)
                    ys = int((bars['y0'])*map_sz/height)
                    print(bars['x0'], '->', xs,bars['y0'], '->', ys )  
                    if xs == map_sz :
                        xs-=1
                    if ys == map_sz :
                        ys-=1
                    if xs < map_sz and ys < map_sz :
                        canvas[ys,xs] = 1
                    else: 
                        print('ERROR', xs, ys, map_sz, f)
                        print(lod)
                    ## top right
                    # xs = int(((bars['x0']- x0)+bars['width'])*map_sz/width)
                    # ys = int(((bars['y0']- y0)+bars['height'])*map_sz/height)           
                    xs = int(((bars['x0'])+bars['width'])*map_sz/width)
                    ys = int(((bars['y0'])+bars['height'])*map_sz/height)           
                    print(bars['x0'], '->', xs, bars['y0'], '->', ys )  
                    if xs == map_sz :
                        xs-=1
                    if ys == map_sz :
                        ys-=1
                    if xs < map_sz and ys < map_sz :
                        canvas[ys,xs] = 1
                    else: 
                        print('ERROR', xs, ys, map_sz, f)
                        print(lod)
            else :
                raise ValueError('Chart Type Not Found')
            if canvas.sum() == 0 :
                raise ValueError('HM sum 0 Found')    
            # canvas =  gaussian_filter(canvas, 2)
            np.save(osp.join(sd,f[:-5]), canvas)
            print('Save success')
            print("*"*80)


'''
    js_obj = json.load(open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/line/PMC2674413___1297-9686-41-32-6.json', 'r'))
    ln_data = js_obj['task6']['output']['visual elements']['lines']
    js_obj = json.load(open('/home/sahmed9/Documents/data/charts/ICPR_ChartCompetition2020_AdobeData/Task-level-JSONs/JSONs/vbox/59253.json', 'r'))
    box_data = js_obj['task6']['output']['visual elements']['boxplots']
    height, width, x0, y0 = js_obj['task6']['input']['task4_output']['_plot_bb'].values()
    js_obj['task6']['input']['task4_output']['_plot_bb']
    ## Decode from heatmap 
    ### np.unravel_index(np.argsort(canvas.ravel()), canvas.shape)

    c = countIslands(canvas, 0.04)
    c_ = torch.zeros(128,128)
    for t in c : 
        for p in t : 
                c_[p[0]][p[1]] = 1

    plt.imshow(c_); plt.show()


    plt.imshow(np_img); plt.axis(False); plt.show()
    
'''

def run_test():
    arg = {
        'Batch_size' : 1,
        'Epochs' : 20,
        'lr' : 0.00001,
        'cache_dir' : './',
        'cache_file' : 'AllPMC_PointDetectorHgNet_8b',
        'load_file': '/home/sahmed9/reps/DeepRule/utils/cache/saves/_ep_19_model_AllPMC_PointDetectorHgNet_8b.pth',
        'op_dir': '/home/sahmed9/reps/DeepRule/utils/cache/hms/8b/',
        'n_class' : 2, 
        'loss' : 'wtMSE-bce-ce',  
        'model' : 'hg_3_1', 
        'cuda': True
    }

    print(arg)
    if not osp.isdir(arg['op_dir']) :
        os.mkdir(arg['op_dir'])

    if arg['model'] == 'hg_3_1' :
        model  = hg(num_stacks=3, num_blocks=1, num_classes=arg['n_class'])
    else :
        model = Point_Detector_Lenovo_FCN(n_classes=arg['n_class'], loss=arg['loss'])
    if arg['cuda'] :
        model = model.cuda()
    print(model)
    test_ds = PMC_line('test')
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    # writer = SummaryWriter(log_dir='./runs/'+arg['cache_file'])
    with open(arg['load_file'], "rb") as f:
        params = torch.load(f)
        optimizer.load_state_dict(params['optim_wt_'])
        model.load_state_dict(params['mod_wt_'])
    print('Model + Optim loaded success', arg['load_file'])

    threshold1 = 50
    threshold2 = 100
    threshold3 = 200
    threshold4 = 300
    threshold5 = 400
    with torch.no_grad() :
        model.eval()
        tit = iter(test_dataloader)
        
        for t in tit : 
            img, gt = t 
            pb, x_axs, y_axs, og_sz, fl_nm = gt 
            print('*'*80)
            print(pb, og_sz, fl_nm)
            im = img.cuda()
            hm  = model(im)

            # hm = F.interpolate(hm, size=og_sz, mode='bilinear')
            ## x, y -> x+ width, y+height
            shp = [(pb['x0'], pb['y0']), (pb['x0']+pb['width'], pb['y0']+pb['height'])]
            print('shp', shp)
            if arg['loss'] == 'wtMSE-bce-ce' :
                for ix, h in enumerate(hm) : 
                    print(ix, 'h in hm', h.shape)
            else :
                print('hm', hm.shape)

            # np.save(osp.join(arg['op_dir'], fl_nm[0]), (hm[0][0][0].detach().cpu().numpy(), hm[1][0][1].detach().cpu().numpy() ))
            np.save(osp.join(arg['op_dir'], fl_nm[0]), (hm[0][0].detach().cpu().numpy(), hm[1][0].detach().cpu().numpy(), hm[2][0].detach().cpu().numpy() ))
            print('Saved', osp.join(arg['op_dir'], fl_nm[0]))
            # exit()
            continue
            ## NCHW
            # get inner plot area
            hm_c = hm[:, :, shp[0][1]:shp[1][1] , shp[0][0]:shp[1][0]]
            canvas = hm_c.detach().numpy()

            islands = countIslands(canvas[0][0], th=0.03)
            lines = getLines(islands, th=0)

            ## Take k top points -> better to take Connected componenets
            # x_, y_ = np.unravel_index(np.argsort(canvas.ravel()), canvas.shape)
            # # get topk points
            # x_1, y_1 = x_[-threshold1:], y_[-threshold1:]
            # x_2, y_2 = x_[-threshold2:], y_[-threshold2:]
            # x_3, y_3 = x_[-threshold3:], y_[-threshold3:]
            # x_4, y_4 = x_[-threshold4:], y_[-threshold4:]
            # x_5, y_5 = x_[-threshold5:], y_[-threshold5:]


            ## group points into lines ?? 

            ## get value for each point per line 
            


## connectred componenets on 2d map = mat
def cc(mat):
    M, N = len(mat), len(mat[0])
    thodl = 0
    visit = [[False for _ in range(N)] for _ in range(M)]
    def dfs(i, j, visited, cmp) :
        # print('in dfs', i, j, cmp, visited[i][j])
        if visited[i][j] :
            return cmp, visited 
        else : 
            visited[i][j] = True
            cmp.append((i, j))
        if j+1 < M:
            if mat[i][j+1] > thodl :
                cmp, visited = dfs(i, j+1, visited, cmp)  
        if j+1 < M and i+1 <N:
            if mat[i+1][j+1] > thodl :
                cmp, visited = dfs(i+1, j+1, visited, cmp)  
        if j-1 >= 0 : 
            if mat[i][j-1] > thodl :
                cmp, visited = dfs(i, j-1, visited, cmp) 
        if j-1 >= 0 and i-1 >=0 : 
            if mat[i-1][j-1] > thodl :
                cmp, visited = dfs(i-1, j-1, visited, cmp) 
        if i-1 >= 0 :
            if mat[i-1][j] > thodl :            
                cmp, visited = dfs(i-1, j, visited, cmp) 
        if i-1 >= 0 and j+1 < M:
            if mat[i-1][j+1] > thodl :            
                cmp, visited = dfs(i-1, j+1, visited, cmp) 
        if i+1 < N :
            if mat[i+1][j] > thodl :
                cmp, visited = dfs(i+1, j, visited, cmp) 
        if i+1 < N  and j-1 >=0:
            if mat[i+1][j-1] > thodl :
                cmp, visited = dfs(i+1, j-1, visited, cmp) 
        return cmp, visited
    comps = []
    for r in range(M) :
        for c in range(N):
            cmp = []
            if not visit[r][c] and mat[r][c] > thodl :
                cmp, visit = dfs(r, c, visit, cmp)
            if len(cmp) > 0 :
                comps.append(cmp)
    return comps



def getLines(islands, th ) :

    def getCenter(isl) :
        isl = sorted(isl, key=lambda k:k[0])
        ctrs = set()
        ctrs.add(isl[int(len(isl)/2)], isl[int(len(isl)/2)+1], isl[int(len(isl)/2)-1])

        isl = sorted(isl, key=lambda k:k[1])
        if len(isl) & 1 == 1 :
            y_ctr =  isl[int(len(isl)/2)+1]
        else : 
            y_ctr =  (isl[int(len(isl)/2)+1], isl[int(len(isl)/2)+1])
        
        
    lines = []
    for isle in islands :
        if len(isle) == 1 : 
            center = isle[0]
        else :
            center = getCenter(isle)

    return lines 



def countIslands(mat, th=0):
    print('In countIslands', mat.shape)
    M, N = len(mat), len(mat[0])
    vis = [[False for i in range(N)] for i in range(M)]
    res = 0
    comps = []
    thold = th
    def isSafe(mat, i, j, vis):
        return ((i >= 0) and (i < N) and
                (j >= 0) and (j < M) and
            (mat[i][j]>thold and (not vis[i][j])))    
    def BFS(mat, vis, si, sj, cmp):
        row = [-1, -1, -1,  0, 0,  1, 1, 1]
        col = [-1,  0,  1, -1, 1, -1, 0, 1]
        q = deque()
        q.append([si, sj])
        vis[si][sj] = True
        cmp.append((si, sj))
        while (len(q) > 0):
            temp = q.popleft()
            i = temp[0]
            j = temp[1]
            for k in range(8):
                if (isSafe(mat, i + row[k], j + col[k], vis)):
                    vis[i + row[k]][j + col[k]] = True
                    cmp.append((i + row[k], j + col[k]))
                    q.append([i + row[k], j + col[k]])
        return cmp
    for i in range(M):
        for j in range(N):
            comp = []
            if (mat[i][j] > thold and not vis[i][j]):
                comp = BFS(mat, vis, i, j, comp)
                res += 1
            if len(comp) > 0 :
                comps.append(comp)
    print('Totoal componenets found = ', res)
    return comps







import sys

if __name__ == "__main__":
    if sys.argv[1] == '1' :
        print('Run Train')
        run_train()
    elif sys.argv[1] == '2' :
        print('Run Test')
        run_test()

# jj = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/json_type/line/PMC1166547___1471-2156-6-26-4.json'
# jj = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/json_type/line/PMC3277264___fgene-03-00012-g001.json'
# imf = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/images/PMC3277264___fgene-03-00012-g001.jpg'

# hm = '/home/sahmed9/Documents/data/tash/PMC514554___1471-2458-4-32-2.npy'
# imf = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/line/PMC514554___1471-2458-4-32-2.jpg'

# imf = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/images/PMC1166547___1471-2156-6-26-4.jpg'

def visualise_heat_img():
    ## Test 
    img_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/images/'
    json_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/json_type/line/'
    # hm_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/hms/1/'
    # op_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/oput/1'
    # hm_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/hms/2/'
    # op_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/oput/2/'
    # hm_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/hms/7a/'
    # op_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/oput/7a/'
    # hm_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/hms/8a/'
    # op_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/oput/8a/'
    # hm_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/hms/8b/'
    # op_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/oput/8b/'
    # ## Train Ground Truth 
    hm_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/hms/9/'
    op_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/oput/9/'
    # ## Train Ground Truth 
        # img_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/line/'
        # hm_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/line_mask/'
        # op_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/gt/'
    if not osp.isdir(op_dir) :
        os.mkdir(op_dir)
    for hm in os.listdir(hm_dir) :
        ## image
        fig = plt.figure(figsize=(30, 14))
        fig.add_subplot(4,5,1)
        img = plt.imread(osp.join(img_dir, hm[:-3]+'jpg'))
        plt.title("Input Chart")
        plt.imshow(img)
        ## LAbel;
        fig.add_subplot(4,5,2)
        canvas = torch.zeros((128,128))
        print(img.shape)
        ims = img.shape
        height, width = ims[0], ims[1]
        print('hm', hm)
        print('hm', hm[:-3])
        print('hm', hm[:-3]+'json')
        f =  hm[:-3]+'json'
        print('f', f)
        lbl_file = osp.join(json_dir, f)
        lbl_file = open(lbl_file, 'r')
        js_obj = json.load(lbl_file)
        ln_data = js_obj['task6']['output']['visual elements']['lines']
        pts = []
        def interpolt(a, x1,y1, x2,y2, p=10) :
            if x1 >=128 or x2 >=128 or y1 >=128 or y2 >=128 :
                return a   
            r = max(x2, x1) - min(x2, x1)
            if r != 0 : 
                # print('x', round(r,3), round(r/p, 3))
                z = interp1d([x1,x2],[y1, y2],fill_value="extrapolate")
                xs = np.arange(min(x2, x1), max(x2, x1), r/p)
                ys = z(xs)
                for ix, x in enumerate(xs):
                    a[int(ys[ix]), int(x)] += 1
            else :
                r = max(y2, y1) - min(y2, y1)
                if r != 0 : 
                    # print('y', round(r, 3), round(r/p, 3))
                    z = interp1d([y1, y2], [x1,x2], fill_value="extrapolate")
                    ys = np.arange(min(y2, y1), max(y2, y1), r/p)
                    xs = z(ys)
                    for iy, y in enumerate(ys):
                        a[int(y), int(xs[iy])] += 1
            return a     
        for l in ln_data : 
            for pt in l : 
                pts.append(pt['x'])
                pts.append(pt['y'])
                xs = int(pt['x']*128/width)
                ys = int(pt['y']*128/height)
                if xs >= 128 or ys>=128: 
                    print(xs, ys, '||', pt['x'], pt['y'])                    
                x_ = xs if xs < 128 else 127
                y_ = ys if ys < 128 else 127
                canvas[y_,x_] = 1
        for ln in ln_data : 
            for ix, pt_ in enumerate(ln) :
                px128 = pt_['x']*map_sz/width; py128 =pt_['y'] *map_sz/height
                if ix >0 :
                    ppx128 = ln[ix-1]['x']*map_sz/width; ppy128 =ln[ix-1]['y'] *map_sz/height
                    canvas = interpolt(canvas, ppx128, ppy128, px128, py128)   
        canvas =  gaussian_filter(canvas, 1)
        plt.title("GT Heatmap BCE/MSE")
        plt.imshow(canvas)
        print('*'*80)
        fig.add_subplot(4,2,3)
        canvas[canvas >=1] =1
        canvas[canvas <1] = 0 
        plt.title("GT Heatmap Classification")
        plt.imshow(canvas)
        print('*'*80)
        ### Predictions 
        pred = np.load(osp.join(hm_dir, hm))
        print('got pred', pred.shape)
        hmx = 6
        for hm in pred :
            hm = hm.squeeze(0).shape
            fm_mse = hm[0:2, :, :]
            fm_bce = hm[3, :, :]
            fm_ce  = torch.softmax(torch.from_numpy(hm[3:5, :, :]),dim=1)
            hm_a = F.sigmoid(torch.from_numpy(fm_mse[0])).numpy()
            hm_b = F.sigmoid(torch.from_numpy(fm_mse[1])).numpy()
            fig.add_subplot(4,2,hmx)
            hmx+=1
            plt.title("Pred Heatmap MSE probablity c1")
            plt.imshow(hm_a)
            print('->', hm, np.sum(hm_))       
            fig.add_subplot(4,2,hmx)
            hmx+=1
            plt.title("Pred Heatmap MSE probablity c2")
            plt.imshow(hm_b)
            print('->', hm, np.sum(hm_))       
            ## BCE
            hm_2a = F.sigmoid(torch.from_numpy(fm_bce)).numpy()
            # hm_2b = F.sigmoid(torch.from_numpy(hm_2[1])).numpy()
            fig.add_subplot(4,2,hmx)
            hmx+=1
            plt.title("Pred Heatmap BCE")
            plt.imshow(hm_2a)
            print('->', hm_2a.shape, np.sum(hm_2a))       
            # fig.add_subplot(4,2,6)
            # plt.title("Pred Heatmap probablity c2")
            # fig.add_subplot(4,2,hmx)
            # hmx+=1
            # print('->', hm_2b.shape, np.sum(hm_2b))       
            ## CE
            fig.add_subplot(4,2,hmx)
            hmx+=1
            plt.title("Pred Heatmap classification c1")
            plt.imshow(fm_ce[0])
            # print('-> hm2', np.sum(hm_3[0]))
            fig.add_subplot(4,2,hmx)
            hmx+=1
            plt.title("Pred Heatmap classification c2")
            plt.imshow(fm_ce[1])
            # print('->', hm_3.shape, np.sum(hm_3[1]))       
            #
            plt.savefig(osp.join(op_dir, hm[:-3]+'jpg'))
            # break





def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2
    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue
        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue
            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        # finish
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)


def verify_mask():
    dirss = ['/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/all_pmc_mask/', '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/synth_5_mask']
    dirss = ['/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/all_pmc_mask_plotbb/', '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/synth_5_mask_plotbb']
    s = []
    for dirs in dirss : 
        folders = os.listdir(dirs)
        for f in folders : 
            print('\n FOLDER', f)
            ms = os.listdir(osp.join(dirs, f))
            c = 0
            for m in ms : 
                lbl = np.load(osp.join(dirs, f, m))
                s.append(np.sum(lbl))
                if np.sum(lbl) == 0 :
                    c+=1
            print('zero found {}/{}'.format(c, len(ms)))
                    # print(osp.join(dirs, f, m))
    return s 



    hm_, hm_2, hm_3  = np.load('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/hms/8a/PMC2292166___1471-2458-8-80-1.npy')
    hm_, hm_2, hm_3  = np.load('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/hms/8a/PMC2464733___pgen.1000137.g012.npy')

    hm_a = F.sigmoid(torch.from_numpy(hm_[0])).numpy()
    hm_b = F.sigmoid(torch.from_numpy(hm_[1]))
    hm_2a = F.sigmoid(torch.from_numpy(hm_2[0]))
    hm_2b = F.sigmoid(torch.from_numpy(hm_2[1]))
    hm_3 = F.softmax(torch.from_numpy(hm_3), dim=0)

    plt.imshow(hm_a); plt.show()

    hm_a = F.sigmoid(torch.from_numpy(hm_[0])).numpy()
    for i in range(10000) :
        hm_a[hm_a>= 0.85*hm_a.max()] = hm_a.max()
        hm_a[hm_a<= 0.15*hm_a.max()] = hm_a.min()
        hm_a = gaussian_filter(hm_a, 0.1)

    for i in range(1000) :
        hm_a[hm_a>= 0.8*hm_a.max()] = hm_a.max()
        hm_a[hm_a<= 0.2*hm_a.max()] = hm_a.min()
        hm_a = gaussian_filter(hm_a, 0.05)

    for i in range(500) :
        hm_a[hm_a>= 0.4*hm_a.max()] = hm_a.max()
        hm_a[hm_a<= 0.35*hm_a.max()] = hm_a.min()
        hm_a = gaussian_filter(hm_a, 0.01)

    from sklearn.cluster import KMeans
    x_, y_ = np.unravel_index(np.argsort(hm_a.ravel()), hm_a.shape)
    pts = []
    for x, y in zip(x_[-200:], y_[-200:]) :
        pts.append([x,y])

    X = np.array(pts)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

    lbls = set(kmeans.labels_)
    lns = {}

    for l in lbls :
        lns.update({l:[]})

    for ix, l in enumerate(kmeans.labels_) :
        lns[l].append({"x": pts[ix][0],"y": pts[ix][1]})

    x0 = [_['x'] for _ in lns[0]]
    y0 = [_['y'] for _ in lns[0]]

    x1 = [_['x'] for _ in lns[1]]
    y1 = [_['y'] for _ in lns[1]]
'''
js_obj = json.load(open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/line/PMC2674413___1297-9686-41-32-6.json', 'r'))
im = Image.open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/line/PMC2674413___1297-9686-41-32-6.jpg')
hm = np.load('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/all_pmc_mask_plotbb_128/line/PMC2674413___1297-9686-41-32-6.npy')
sz = 128
a = np.zeros((sz, sz))
b = np.zeros((sz, sz))
c = np.zeros((sz, sz))
w, h = im.size
for ln in ln_data : 
    for ix, pt in enumerate(ln) :
        if ix >0 :
            ppx128 = ln[ix-1]['x']*sz/w; ppy128 =ln[ix-1]['y'] *sz/h
        else :
            ppx128 = 0; ppy128 =0
        px128 = pt['x']*sz/w; py128 =pt['y'] *sz/h
        if ix+1<len(ln) :
            npx128 = ln[ix+1]['x']*sz/w; npy128 =ln[ix+1]['y'] *sz/h
        else :
            npx128 = 0; npy128 =0
        for x in range(sz) :
            for y in range(sz) :
                    valb = round(math.exp(-((x-px128)**2 + (y-py128)**2))/4, 2)
                    # print('valb', valb)
                    px = (ppx128 + npx128 + px128)/3; py = (ppy128 + npy128 + py128)/3
                    vala = round(math.exp(-((x-px)**2 + (y-py)**2))/4, 2)
                    vala = math.exp(-((x-px128)**2 + (y-py128)**2))/4 
                    # print('vala', vala)
                    px = 0.2*ppx128 + 0.2*npx128 + 0.6*px128; py = 0.2*ppy128 + 0.2*npy128 + 0.6*py128
                    valc = round(math.exp(-((x-px)**2 + (y-py)**2))/4, 2)
                    if vala > 0 :
                        a[y, x] += vala        
                    if valb > 0 :
                        b[y, x] += valb        
                    if valc > 0 :
                        c[y, x] += valc        

f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(gaussian_filter(lb, 0.3))
axarr[0,1].imshow(gaussian_filter(lb, 1))
axarr[1,0].imshow(b)
axarr[1,1].imshow(a)
plt.axis('off')
plt.show()
plt.imshow(lb);plt.imshow(b);plt.imshow(a);plt.imshow(c); plt.show()

plt.imshow(a); plt.show()

for x in range(128) :
    for y in range(128) :
            val = round(math.exp(-((x-px128)**2 + (y-py128)**2))/4, 5)
            if val > 0 :
                print(x, y, val)
            # a[r, c] = math.exp(-((r-py128**2)+ (c-px128**2))/4)




def gauss(a, px128, py128, ppx128, ppy128, npx128, npy128, sz):
    for x in range(sz) :
        for y in range(sz) :
            vala = math.exp(-((x-px128)**2 + (y-py128)**2))/4
            vala += math.exp(-((x-ppx128)**2 + (y-ppy128)**2))/4
            vala += math.exp(-((x-npx128)**2 + (y-npy128)**2))/4
            if vala > 0 :
                a[y, x] += vala    

def interpolt(a, x1,y1, x2,y2, p=10) :
    # print(x1,y1, x2,y2)
    z = interp1d([x1,x2],[y1, y2])
    # print(z)
    r = max(x2, x1) - min(x2, x1)
    # print(r)
    xs = np.arange(min(x2, x1), max(x2, x1), r/p)
    # print(xs)
    ys = z(xs)
    # print(ys)
    for ix, x in enumerate(xs):
        a[int(ys[ix]), int(x)] += 1


sz = 128
a = np.zeros((sz, sz))
for ln in ln_data : 
    for ix, pt in enumerate(ln) :
        px128 = pt['x']*sz/w; py128 =pt['y'] *sz/h
        if ix >0 :
            ppx128 = ln[ix-1]['x']*sz/w; ppy128 =ln[ix-1]['y'] *sz/h
            interpolt(a, ppx128, ppy128, px128, py128)
        else :
            ppx128 = 0; ppy128 =0
        if ix+1<len(ln) :
            npx128 = ln[ix+1]['x']*sz/w; npy128 =ln[ix+1]['y'] *sz/h
        else :
            npx128 = 0; npy128 =0
        

for ln in ln_data : 
    for ix, pt in enumerate(ln) :
        px128 = pt['x']*sz/w; py128 =pt['y'] *sz/h
        if ix >0 :
            ppx128 = ln[ix-1]['x']*sz/w; ppy128 =ln[ix-1]['y'] *sz/h            
        else :
            ppx128 = 0; ppy128 =0
        if ix+1<len(ln) :
            npx128 = ln[ix+1]['x']*sz/w; npy128 =ln[ix+1]['y'] *sz/h
        else :
            npx128 = 0; npy128 =0
        gauss(a, px128, py128, ppx128, ppy128, npx128, npy128, sz)


for ln in ln_data : 
    for ix, pt in enumerate(ln) :
        # print(ix)
        px128 = pt['x']*sz/w; py128 =pt['y'] *sz/h    
        # print(px128, py128, pt['x'], pt['y'], a[int(py128), int(px128)] )
        a[int(py128), int(px128)] += len(ln_data)
        # print(int(px128), int(py128), pt['x'], pt['y'], a[int(py128), int(px128)] )

f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(np.array(im))
axarr[0,1].imshow(hm)
axarr[1,0].imshow(a)
axarr[1,1].triplot(points[:,1], points[:,1], tri.simplices)
axarr[1,1].plot(points[:,1], points[:,0], 'o')

plt.axis('off')
plt.tight_layout(True)
plt.show()


px, py = np.unravel_index(np.argsort(a.ravel()), a.shape)
for i in range(1,53) :
    print(i, ':::', px[-i], py[-i], px[-i]/0.2133, py[-i]/0.215488, a[px[-i], py[-i]])

'''
def so(vv):
    hm = np.load('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/hms/9/PMC2672343___ijerph-06-00232f3.npy')
    x = hm[2, 0, 1, :, :]
    x = F.sigmoid(torch.from_numpy(x)).numpy()
    x_, y_ = np.unravel_index(np.argsort(x.ravel()), x.shape)
    pts = []
    pred = torch.zeros(128,128)
    for x, y in zip(x_[-vv:], y_[-vv:]) :
            pts.append([x,y])
    for p in pts : 
        pred[p[0], p[1]] = 1
    plt.imshow(pred); plt.show()



kk_ = np.load('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/hms/9/PMC2672343___ijerph-06-00232f3.npy')
js_obj = json.load(open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/json_type/line/PMC2672343___ijerph-06-00232f3.json'))
ln_data = js_obj['task6']['output']['visual elements']['lines']
img = PIL.Image.open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/oput/8a/PMC2672343___ijerph-06-00232f3.jpg')

width, height = img.size
gt_points_ = []
for ix, ln in enumerate(ln_data) : 
    gt_points = []
    for pt_ in ln : 
        xs = int((pt_['x'])*128/width)
        ys = int((pt_['y'])*128/height)
        x_norm = pt_['x']/width
        y_norm = pt_['y']/height
        gt_points.append([[xs, ys], [x_norm, y_norm]])
    gt_points_.append(gt_points)

gt_points_ =[gt_points*3]
        

hm1, hm2, hm3 = [torch.from_numpy(kk_[0]), torch.from_numpy(kk_[1]), torch.from_numpy(kk_[2])]
hm1, hm2, hm3 = [torch.randn(3, 256,128,128), torch.randn(3, 256,128,128),torch.randn(3, 256,128,128)]
chnl_ = hm1.shape[1]*3
m1 = nn.Conv2d(chnl_, 1, 1)
m2 = nn.Conv2d(chnl_, 1, 1)
m3 = nn.Conv2d(chnl_, 1, 1)
m4 = nn.Conv2d(chnl_, hm1.shape[1], 1)
node_emb = nn.Linear(258, 128)
maps = []
for hix in range(hm1.shape[0]) : 
    maps.append(hm1[hix])
    maps.append(hm2[hix])
    maps.append(hm3[hix])

maps = torch.stack(maps)
print(maps.shape)

maps = maps.reshape(int(maps.shape[0]/3), -1, maps.shape[-2], maps.shape[-1])
print(maps.shape)

maps_alpha = F.softmax(m1(maps)*m2(maps), dim=1)
print(maps_alpha.shape)

maps_beta = F.sigmoid(maps_alpha * m3(maps))
print(maps_beta.shape)

maps *= maps_beta
print(maps.shape)

maps = m4(maps)
print(maps.shape)
nodes = []
for batch in range(maps.shape[0]) :
    op = []
    for _ in gt_points_ :
        print('_', _)
        print('_[0]', _[0],  _[0][0], _[0][1], graphs)
        fp = maps[graphs, :,  _[0][0], _[0][1]]
        print('fp', fp.shape)
        fp_ = fp.new([_[1][0], _[1][1]])
        print('fp_', fp_.shape)
        raw = torch.cat((fp, fp_))
        print('raw', raw.shape)
        # n_e = node_emb(raw)
        op.append(raw)  
        print('op_', op[-1].shape)
    nodes.append(op)