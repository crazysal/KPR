import torch
from torchvision.utils import save_image, make_grid
import os.path as osp
import numpy as npy 
import os 
import gc 
import pickle
# plt.rcParams["savefig.bbox"] = 'tight'

import json 

class Runner:
    def __init__(self, **kwd):
        # dataloaders
        self.config = kwd['config']
        self.train_dataset = kwd['train_ds']
        self.train_dataloader = kwd['train_dl']
        self.val_dataset = kwd['val_ds']
        self.val_dataloader = kwd['val_dl']
        self.model = kwd['model']
        self.criterion = kwd['criterion']
        self.optimizer = kwd['optim']
        self.wt = kwd['writer']

    def train(self, ep):
        print('*'*80)
        print('Start Train Epoch', ep)
        print('*'*80)
        tr_iter = iter(self.train_dataloader)
        running_loss = []
        running_accuracy = []
        for ix, data in enumerate(tr_iter) :
            gc.collect()
            image, target, el, pt, ptn, sz = data
            # print('in train')
            # print(image.shape, target.shape, len(el), len(el[0]),  len(pt), len(pt[0]),  len(ptn), len(ptn[0]))
            if self.config.cuda is True : 
                image, target = image.cuda(), target.cuda()
            self.optimizer.zero_grad()
            # output = self.model(image,el, pt, ptn, self.config.split)
            if ep >= 3 : 
                decoder_loss = True
            else :                
                decoder_loss = False
            # output = self.model(image,el, pt, ptn, self.config.split, decoder_loss, sz)

            pnl = []
            pl128 = []
            point_list, pt_idx = pt 
            if sz is not None : 
                for idx, osz in enumerate(sz) : 
                    ln_ = []
                    ln_128 = []
                    points = point_list[idx]
                    for pt in points :
                        pt0, pt1 = pt
                        pt0/=osz[0]
                        if pt0 >=1 :
                            print('::: ERROR ::: x', pt0, osz) 
                            print('::: ', pt, osz) 
                            pt0 = 0.99
                        if pt0 <0 :
                            print('::: ERROR ::: x', pt0, osz) 
                            print('::: ', pt, osz) 
                            pt0 = 0
                        pt0128 = pt0*128
                        pt1/=osz[1]
                        if pt1>=1: 
                            print('::: ERROR ::: y', pt1, osz) 
                            print('::: ', pt, osz) 
                            pt1 = 0.99
                        if pt1<0: 
                            print('::: ERROR ::: y', pt1, osz) 
                            print('::: ', pt, osz) 
                            pt1 = 0
                        pt1128 = pt1*128
                        ln_.append((pt0, pt1))
                        ln_128.append((pt0128, pt1128))
                    pnl.append(ln_)
                    pl128.append(ln_128)
                point_norm_list = pnl
            output = self.model(image, (pnl, pl128, pt_idx), self.config.split, sz)
            
            tr_l = self.criterion(output, target, self.config.split, self.wt, ep, ix)
            running_loss.append(tr_l.item())
            running_accuracy.append(1)
            tr_l.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            del output
            # if ix > 4 :
                # exit()

        return running_loss, running_accuracy
    
    def validate(self, ep):
        print('#'*80)
        print('Start Val Epoch', ep)
        print('#'*80)
        val_iter = iter(self.val_dataloader)
        running_loss = []
        running_accuracy = [1]
        
        for ix, data in enumerate(val_iter) :
            image, target, el, pt, ptn, sz = data
            if self.config.cuda is True : 
                image, target = image.cuda(), target.cuda()
            # output =  self.model(image,el, pt, ptn, self.config.split)
            # output =  self.model(image,el, pt, ptn, 'val', True, sz)
            # output = self.model(image, pt, self.config.split, sz)

            pnl = []
            pl128 = []
            point_list, pt_idx = pt 
            if sz is not None : 
                for idx, osz in enumerate(sz) : 
                    ln_ = []
                    ln_128 = []
                    points = point_list[idx]
                    for pt in points :
                        pt0, pt1 = pt
                        pt0/=osz[0]
                        if pt0 >=1 :
                            print('::: ERROR ::: x', pt0, osz) 
                            print('::: ', pt, osz) 
                            pt0 = 0.99
                        if pt0 <0 :
                            print('::: ERROR ::: x', pt0, osz) 
                            print('::: ', pt, osz) 
                            pt0 = 0
                        pt0128 = pt0*128
                        pt1/=osz[1]
                        if pt1>=1: 
                            print('::: ERROR ::: y', pt1, osz) 
                            print('::: ', pt, osz) 
                            pt1 = 0.99
                        if pt1<0: 
                            print('::: ERROR ::: y', pt1, osz) 
                            print('::: ', pt, osz) 
                            pt1 = 0
                        pt1128 = pt1*128
                        ln_.append((pt0, pt1))
                        ln_128.append((pt0128, pt1128))
                    pnl.append(ln_)
                    pl128.append(ln_128)
                point_norm_list = pnl
        
            output = self.model(image, (pnl, pl128, pt_idx), 'val', sz)
            v_l = self.criterion(output, target,'val', self.wt, ep, ix)
            running_loss.append(v_l.item())
            # running_accuracy.append(1)
        return running_loss, running_accuracy
    
    def test(self):
        print('$'*80)
        print('Start Test')
        print('$'*80)
        val_iter = iter(self.val_dataloader)
        self.model.eval()
        with torch.no_grad() :
            for ix, data in enumerate(val_iter) :
                # print(data)
                # image, op_file, el, pt, ptn = data['image'], data['trg'], data['el'], data['pt'], data['pt_n']
                image, op_file, el, pt, ptn, sz = data
                print('*'*80)
                print(image.shape, op_file, len(el[0]), len(pt[0]), len(ptn[0]))
                if self.config.cuda is True : 
                    image= image.cuda()
                ## 9
                # op_ = self.model(image, el, pt, ptn, 'test')
                ## 11 onwards
                # op_ = self.model(image, el, pt, ptn, 'test', True)
                ## 13 

                op_ = self.model(image, None, 'test', sz)
                # print('output', op_)
                # # op_ = op_[0] + op_[1]
                # for p in op_ :
                #     print(p.shape)
                # if isinstance(op_ , list) :
                #     print(len(op_))
                #     for o in op_ :
                #         print(o.shape)
                # else :
                #         print(op_.shape)
                # exit()
                # self.save_heatmap(op_, op_file, True, sz[0])
                # 9 
                # self.save_heatmap(op_, op_file, False, sz[0])
                ## 13
                # self.save_heatmap(op_['op'], op_file, 'pool_net', sz[0])
                self.save_heatmap(op_['op'], op_file, 'pool_fcn_net', sz[0])
                # exit()
    # def accuracy(self, op, trg) :
    #     return torch.tensor([True]) if op == torch.tensor([trg]) else False
    

    def update_lr(self, ep, lr) :
        lr/=10
        print('Epoch {} LR Updated from {} to {}'.format(ep, lr*10, lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr
    def save(self, sv) :
        # if self.config.isDP :
            # params = {'mod_wt_' : self.model.module.state_dict(), 'optim_wt_' : self.optimizer.state_dict() }
        # else:
        params = {'mod_wt_' : self.model.state_dict(), 'optim_wt_' : self.optimizer.state_dict() }
        with open(sv, "wb") as f:
            torch.save(params, f)
        # torch.save(params, f, _use_new_zipfile_serialization=False)
        print('Model + Optim saved success', sv)
    def load(self) :
        with open(self.config.load_file, "rb") as f:
            params = torch.load(f)
            if self.optimizer is not None :
                self.optimizer.load_state_dict(params['optim_wt_'])
                print('Optim loaded success', self.config.load_file)
            
            # parmas_ = dict()
            # for p in params['mod_wt_'] :
                # parmas_.update({'m1.'+p[10:]:params['mod_wt_'][p]})
            # _ = self.model.load_state_dict(parmas_, strict=False)
            
            _ = self.model.load_state_dict(params['mod_wt_'], strict=False)
            print('missing_keys=', len(_[0]),'unexpected_keys=', len(_[1]))
            # for ii in _[0] :
            print(_[0])
            print('_'*80)
            # for ii in _[1] :
            print(_[1])
            # self.model.load_state_dict(params['mod_wt_'])
            print('Model loaded success', self.config.load_file)
    def visualize(self, imgs) :
        save_image(make_grid(imgs, nrow=6),self.config['op_file'])
        print('Save Success', self.config['op_file'])
    def save_heatmap(self, op_, op_file, jnos, sz=None) :
        print('jnos', jnos)
        if jnos=='pool_net' : 
            print('lenop', len(op_))            
            hm, em, reg = op_[-1]
            fm_mse = hm[:, 0:2, :, :]
            fm_bce = hm[:, 2, :, :]
            fm_ce  = torch.softmax(hm[:, 3:5, :, :],dim=1)
            print('fm_mse, fm_bce, fm_ce', fm_mse.shape, fm_bce.shape, fm_ce.shape)
            print('em', em.shape)
            print('reg', reg.shape)
            fin = torch.cat((fm_mse.squeeze(0), fm_bce, fm_ce.squeeze(0), em.squeeze(0), reg.squeeze(0)))
            print(fin.shape)
            hms = fin.detach().cpu().numpy()
            print(fin.shape)
            if not osp.isdir(self.config.op_dir) :
                os.mkdir(self.config.op_dir) 
            op_file = osp.join(self.config.op_dir, op_file[0][:-4]+'pkl')
            # npy.save(op_file, hms)
            with open(op_file, 'wb') as f:
                pickle.dump(hms, f)
            print('Save Success', op_file)


        elif jnos=='pool_fcn_net' : 
            print('lenop', len(op_))            
            hm, em, reg = op_
            fm_mse = hm[:, 0:2, :, :]
            fm_bce = hm[:, 2, :, :]
            fm_ce  = torch.softmax(hm[:, 3:5, :, :],dim=1)
            print('fm_mse, fm_bce, fm_ce', fm_mse.shape, fm_bce.shape, fm_ce.shape)
            print('em', em.shape)
            print('reg', reg.shape)
            fin = torch.cat((fm_mse.squeeze(0), fm_bce, fm_ce.squeeze(0), em.squeeze(0), reg.squeeze(0)))
            print(fin.shape)
            hms = fin.detach().cpu().numpy()
            print(fin.shape)
            if not osp.isdir(self.config.op_dir) :
                os.mkdir(self.config.op_dir) 
            op_file = osp.join(self.config.op_dir, op_file[0][:-4]+'pkl')
            # npy.save(op_file, hms)
            with open(op_file, 'wb') as f:
                pickle.dump(hms, f)
            print('Save Success', op_file)





        elif jnos=='transformer' : 
            ### Convert to json format image coord
            op_obj = {"task6": {"output": { "data series": [] ,"visual elements": {"lines": []}}}}
            major_clusters = op_
            width, height = sz[0], sz[1]
            for k in range(len(major_clusters)):
                coord_arr = []
                for j in range(len(major_clusters[k])):
                    coord_arr.append({"x" : major_clusters[k][j][0]*width/128, "y" : major_clusters[k][j][1]*height/128})
                op_obj["task6"]["output"]["visual elements"]["lines"].append(coord_arr)
            
            print('final json', op_obj)
            op_file = osp.join(self.config.op_dir, op_file[0])
            with open(op_file, 'w') as f_:
                json.dump(op_obj, f_)
            print('Saved', op_obj["task6"]["output"]["visual elements"]["lines"])

        elif jnos=='gnn' :
            k = op_['kp']
            r = op_['rel']
            nodes, trg , edge_list = r
            node_feats, cts, sub_comp_idx = nodes
            # edg_class, edge
            node_feats = node_feats.detach().cpu().numpy()
            # print(nodes, trg, edge_list)
            hms = [hm.detach().cpu().numpy() for hm in k]
            # print('\n edg_cls \n')
            edg_cls = [_[1] for _ in edge_list]
            edg_cls = torch.cat(edg_cls)
            edg_cls = edg_cls.detach().cpu().numpy()
            # print(edg_cls.shape)
            edg_emb = [_[0] for _ in edge_list]
            edg_emb = torch.cat(edg_emb)
            edg_emb = edg_emb.detach().cpu().numpy()
            # print('\n edg_emb \n')
            # print(edg_emb.shape)
            hms.append(node_feats)
            hms.append(edg_cls)
            hms.append(edg_emb)

            print('h in hms')
            for h in hms : 
                print(npy.shape(h))
                
            hms = npy.array(hms, dtype=object)
            # # print()
            # exit()
            if not osp.isdir(self.config.op_dir) :
                os.mkdir(self.config.op_dir) 
            op_file = osp.join(self.config.op_dir, op_file[0][:-4]+'.npy')
            npy.save(op_file, hms)
            print('Save Success', op_file)
        else :
            hms = [hm.detach().cpu().numpy() for hm in op_]
            print('h in hms')
            # hms = npy.array(hms)
            # for h in hms : 
                # print(npy.shape(h))
            if not osp.isdir(self.config.op_dir) :
                os.mkdir(self.config.op_dir) 
            op_file = osp.join(self.config.op_dir, op_file[0][:-4]+'pkl')
            # npy.save(op_file, hms)
            with open(op_file, 'wb') as f:
                pickle.dump(hms, f)
            print('Save Success', op_file)




