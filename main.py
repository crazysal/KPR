import argparse
import os
import os.path as osp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from run.train import Runner
from model.hg_net import hg
from model.gnn_net import krp
from model.ptFormer import ptf
# from model.pool_net import poolnet
from model.pool_fcn_net import poolnet
from criterion.loss import Loss
from config.configs import Configs
from dataset.keypoint_dataset import PMC_line

torch.autograd.set_detect_anomaly(True)

def init_model(arg) :
    class DPM(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.dp = nn.DataParallel(m)
        def forward(self, x, e, p, pn, sp):
            x = self.dp(x, e, p, pn, sp)
            return x
    
    class E2E(nn.Module):
        def __init__(self, m1, m2, config):
            super().__init__()
            self.m1 = m1
            self.m2 = m2
            self.config = config
            if arg.model_mode == 'M2' :
                with open(arg.load_file, "rb") as f:
                    params = torch.load(f)
                    parmas_ = dict()
                    for p in params['mod_wt_'] :
                        parmas_.update({p[10:]:params['mod_wt_'][p]})
                _ = self.m1.load_state_dict(parmas_, strict=False)
                print('E2E, missing_keys=', len(_[0]),'unexpected_keys=', len(_[1]))
                print('Model Backbone Only loaded success', self.config.load_file)
        def forward(self, x, edge_list, point_list, point_norm_list, splt_):
            # print('In e2e fwd')
            # if isinstance(x, list) :
            #     for _ in x :
            #         print(_.shape)
            if arg.model_mode == 'M1' :
                if splt_ == 'train' :
                    self.m1.train()
                    return self.m1(x)[0]
                else:
                    self.m1.eval()
                    with torch.no_grad():
                        print('FROM M1 EVAL ')
                        # return self.m1(x)[0]
                        return self.m1(x)
            elif arg.model_mode == 'M2' :
                self.m1.eval()
                with torch.no_grad():
                    x, x_= self.m1(x)                   
                if splt_ == 'train' :
                    self.m2.train()
                    return self.m2(x_, edge_list, point_list, point_norm_list)
                else :
                    with torch.no_grad():
                        return self.m2(x_, edge_list, point_list, point_norm_list)
            elif arg.model_mode =='M1M2' :
                if splt_ == 'train' :
                    self.m1.train()
                    self.m2.train()
                    x, x_ = self.m1(x)
                    return {'kp':x, 'rel':self.m2(x_,  edge_list, point_list, point_norm_list)}
                else : 
                    self.m1.eval()
                    self.m2.eval()
                    with torch.no_grad():
                        x, x_ = self.m1(x)
                        return {'kp':x, 'rel':self.m2(x_,  edge_list, point_list, point_norm_list)}

    m1 = hg(num_stacks=arg.stacks, num_blocks=arg.blocks, num_classes=arg.classes)
    m2 = krp(config=arg)
    m = E2E(m1, m2, arg)
    if arg.isDP : 
        m = DPM(m)
    # print('Model Initialized')
    # print(m)
    total_params = 0
    for params in m.parameters():
        num_params = 1
        for x in params.size():
            num_params *= x
        total_params += num_params
    print("total model parameters: {}".format(total_params))
    if arg.cuda is True : 
        m = m.cuda()
    o = torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=arg.lr)
    print('Optim Initialized, LR', arg.lr)
    print(o)
    return m , o


def init_loss(arg) :
    l = Loss()
    print('Loss initialized')
    print(l)
    return l

if __name__ == "__main__": 
    confparser = argparse.ArgumentParser()
    confparser.add_argument('--conf_train', type=str, default=None)
    confparser.add_argument('--conf_val', type=str, default=None)
    confparser.add_argument('--conf_test', type=str, default=None)
    confOpt = confparser.parse_args()
    
    print("Loading config number -------    ", confOpt.conf_train)
    C = Configs()
    tr = None   
    
    ## Init writer
    if confOpt.conf_train is not None and confOpt.conf_val is not None :
        print('Run Train')
        print('_'*80)
        arg_train = C.getConfig(confOpt.conf_train)
        for o in vars(arg_train):
            print(o, "\t", vars(arg_train)[o])
        ## Init Data Set/Loder
        tds = PMC_line(config=arg_train)
        tdco = tds.collate_fn
        tdl = DataLoader(tds, batch_size=arg_train.batch_size, shuffle=True, collate_fn=tdco)
        print('_'*80)

        arg_val = C.getConfig(confOpt.conf_val)
        for o in vars(arg_val):
            print(o, "\t", vars(arg_val)[o])
        vds = PMC_line(config=arg_val)
        vdco = vds.collate_fn
        vdl = DataLoader(vds, batch_size=arg_val.batch_size, shuffle=True, collate_fn=vdco)
        print('_'*80)

        ## Init Model, Optimizer
        # mdl, opt = init_model(arg_train)
        # mdl = ptf(config=arg_train)
        mdl = poolnet(config=arg_train)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, mdl.parameters()), lr=arg_train.lr)
        print(mdl)
        print(opt)
        mdl = mdl.cuda()
        total_params = 0
        for params in mdl.parameters():
            num_params = 1
            for x in params.size():
                num_params *= x
            total_params += num_params
        print('Totoal model params = ', total_params)
        
        ## Init Objective Function
        lss = init_loss(arg_train)
        wt = SummaryWriter(log_dir=osp.join(arg_train.rd, arg_train.cache_dir,'runs', arg_train.cache_file))
        # wt = None
        
        tr = Runner(config=arg_train, \
            train_ds=tds, train_dl=tdl, val_ds=vds, val_dl=vdl, \
            model=mdl , optim=opt, criterion=lss, writer=wt)
        lr = arg_train.lr 
        if arg_train.load_file is not None : 
            tr.load()
        for ep in range(0, arg_train.epoch) :
            trunning_loss, trunning_accuracy = tr.train(ep)
            print('EPOCH :{}, Training_Loss:{}/{}={}, Training_Accuracy:{}/{}={}'.format(ep,sum(trunning_loss), len(trunning_loss), sum(trunning_loss)/len(trunning_loss), sum(trunning_accuracy), len(trunning_accuracy), sum(trunning_accuracy)/len(trunning_accuracy)  ))
            if ep % arg_train.saver == 0 :
                svd = osp.join(arg_train.rd, arg_train.cache_dir,'saves', arg_train.cache_file)
                if not osp.isdir(svd):
                    os.mkdir(svd) 
                else: 
                    print('save dir exists', svd)
                sv = osp.join(svd, '_ep_'+str(ep)+'_.pth')
                tr.save(sv)
            if ep % arg_val.val_ep == 0 :
                vrunning_loss, vrunning_accuracy = tr.validate(ep)
                print('EPOCH :{}, Validation_Loss:{}/{}={}, Validation_Accuracy:{}/{}={}'.format(ep,sum(vrunning_loss), len(vrunning_loss), sum(vrunning_loss)/len(vrunning_loss), sum(vrunning_accuracy), len(vrunning_accuracy), sum(vrunning_accuracy)/len(trunning_accuracy)  ))
            # if ep % arg_train.scheduler == 0 :
            #     lr = tr.update_lr(ep, lr)
            #     if ep == 0 :
            #         arg_train.scheduler = arg_train.epoch -5
            #         print('arg_train.scheduler', arg_train.scheduler)

    if confOpt.conf_test is not None :
        arg_test = C.getConfig(confOpt.conf_test)
        for o in vars(arg_test):
            print(o, "\t", vars(arg_test)[o])
        tsds = PMC_line(config=arg_test)
        tsdl = DataLoader(tsds, batch_size=1, shuffle=False, collate_fn=tsds.collate_fn)
        print('_'*80)
        wt = SummaryWriter(log_dir=osp.join(arg_test.rd, arg_test.cache_dir,'runs', arg_test.cache_file))
        print('Run Test')
                ## Init Model, Optimizer
        # mdl, opt = init_model(arg_test)
        # mdl = hg(num_stacks=arg_test.stacks, num_blocks=arg_test.blocks, num_classes=arg_test.classes)
        # mdl = ptf(config=arg_test)
        mdl = poolnet(config=arg_test)
        mdl = mdl.cuda()
        print(mdl)
        if tr is None : 
            tr = Runner(config=arg_test, \
            train_ds=None, train_dl=None, val_ds=tsds, val_dl=tsdl, \
            model=mdl , optim=None, criterion=None, writer=wt) 
        tr.load()
        # exit()
        tr.test()


# import os 
# import json
# im = os.listdir('/home/sahmed9/reps/KeyPointRelations/data/images/test_images_dave/')
# svd='/home/sahmed9/reps/KeyPointRelations/data/dave_line_json_test/'
# for i in im : 
#     f = i[:-3]
#     data = {"task1": {"input": {},"name": "Chart Classification","output": {"chart_type": "line"}},"task2": {},"task3": {},"task4": {},"task5": {},"task6": {}}
#     with open(svd+f+'json', 'w') as outfile:
#         json.dump(data, outfile)
