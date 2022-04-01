import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def _tranpose_and_gather_feature(feature, ind):
  feature = feature.permute(1, 2, 0).contiguous()  # [C, H, W] => [H, W, C]
  feature = feature.view( -1, feature.size(2))  # [H, W, C] => [H x W, C]
  ind = ind[:, None].expand(ind.shape[0], feature.shape[-1])  # [num_obj] => [num_obj, C]
  feature = feature.gather(0, ind)  # [H x W, C] => [num_obj, C]
  return feature


def point_to_index(points) :
    coord_range = {}
    key = 0
    for x_ in range(128) :
        for y_ in range(128) :
            coord_range.update({(x_, y_):key})
            key+=1
    # print('coord_range', coord_range)
    ind = []
    for ln in points : 
        l = []
        for p in ln : 
            l.append(coord_range[(int(p[0]),int(p[1]))])
        ind.append(torch.tensor(l).long().cuda())
    return ind 

class Loss(nn.Module):
    def __init__(self, **kwd):
        super(Loss, self).__init__()        
        self.rec_loss = F.binary_cross_entropy_with_logits
        self.cls_loss = F.cross_entropy
        self.regr_loss = WeightedMSELoss()
        self.edge_pp_loss = MultiSimilarityLoss()
        self.node_pp_loss = PPLoss()
        self.dec_loss = nn.CrossEntropyLoss(ignore_index=16386)
        self.mse = torch.nn.MSELoss()
        self.cls_t = 1
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

    def forward(self, x, gt_lbl, spl, wt, ep, ix) :
        l = []
        mse_lbl = copy.deepcopy(gt_lbl/gt_lbl.max())
        mse_lbl = [1-mse_lbl, mse_lbl]
        mse_lbl = torch.cat(mse_lbl, dim=1)
        bce_lbl = copy.deepcopy(gt_lbl/gt_lbl.max())
        bce_lbl = torch.sigmoid(bce_lbl)
        ce_lbl = copy.deepcopy(gt_lbl)
        ce_lbl[ce_lbl>=self.cls_t] = 1
        ce_lbl[ce_lbl<1] = 0
        ce_lbl = ce_lbl.long()

        if isinstance (x, list) :
            training_loss = []
            for hm in x :
                fm_mse = hm[:, 0:2, :, :]
                fm_bce = hm[:, 2, :, :]
                fm_ce  = torch.softmax(hm[:, 3:5, :, :],dim=1)
                l1 = self.regr_loss(fm_mse, mse_lbl, self.mse_wt()) 
                ### BCE
                # print('lbl->',  lbl.shape, 'inp ->', (hm[0][:, 0, :, :].unsqueeze(dim=1)).shape)
                l2 = self.rec_loss(fm_bce, bce_lbl, weight=self.bce_wt(bce_lbl))
                ### CE
                l3 = self.cls_loss(fm_ce, ce_lbl, weight=self.ce_wt()) 
                training_loss.append(self.alpha*l1+self.beta*l2 +self.gamma*l3)
                wt.add_scalar(spl +" Loss MSE", l1.item(), (ep+1)*(ix+1))
                wt.add_scalar(spl +" Loss BCE", l2.item(), (ep+1)*(ix+1))
                wt.add_scalar(spl +" Loss CE", l3.item(), (ep+1)*(ix+1))
                print(round(l1.item(), 5), round(l2.item(), 5), round(l3.item(), 5))
            tr_l =  sum(training_loss)/len(x)
        elif isinstance (x, tuple) :
            nodes, trg , edge_list= x 
            print('In loss, nd', nodes.shape)
            print('Total trg', len(trg))

            e_class = []
            for eix, _ in enumerate(edge_list) :
                print(eix, _.shape)
                for e in _ :
                    e_class.append(eix) 

            e_class = torch.tensor(e_class).long().to(nodes.device)
            edg = edge_list
            edg1 = torch.clone(edg)
            # edg2 = torch.clone(edg)
            trg = torch.cat(trg).long()
            print('MS class label', e_class, e_class.shape)
            print('Edge shape', edg.shape)
            print('Trgss catted', trg.shape)
            l1 = self.pp_loss(edg1,e_class )
            l2 = self.cls_loss(torch.softmax(edg, dim=-1), trg.to(edg.get_device()))
            tr_l = l1 + l2
            print(l1, l2, tr_l)
            exit()
        elif isinstance (x, dict) :
            t = x['type']
            if t == 'transformer' :
                hm = x['hm']
                fm_mse = hm[0] 
                tr_l = l1 = self.regr_loss(fm_mse, mse_lbl, self.mse_wt()) 
                print(spl +" Loss MSE", tr_l.item())
                wt.add_scalar(spl +" Loss MSE", l1.item(), (ep+1)*(ix+1))

                if x['decoder_loss'] :
                    # print('decoder loss')
                    deq = x['deq']
                    # print('deq', deq.shape)#, deq)
                    deq = nn.functional.softmax(deq, dim=2)
                    # print('deq soft', deq.shape )#, deq.sum(dim=2))
                    crd = x['cor']
                    # print('crd', crd.shape)#, crd)
                    l = x['dec_label']
                    pnl, point_count = x['pnl']
                    # print('pnl', pnl)
                    # print('point_count', point_count)
                    # print('pnl', pnl)
                    pnl = [torch.tensor(a).float() for s in pnl for a in s ]
                    point_count = [a if a <=40 else 40 for s in point_count for a in s ]
                    # print('point_count', point_count)
                    l = x['dec_label']
                    # print('lbl', l.shape, l)
                    deq = deq.reshape(-1, deq.shape[2])
                    l = l.reshape(-1)
                    # print('reshaped', deq.shape, l.shape)
                    # print(deq)
                    # print(l)
                    l2 = self.dec_loss(deq, l)
                    t = []
                    st_ = 0 
                    for p_idx, p in enumerate(point_count) : 
                        # print(p)
                        if p > 0 :
                            en_ = st_+p
                            # print('en_', en_)
                            pln = pnl[st_:en_]
                            # print(pln)
                            pln = torch.stack(pln).cuda()

                            st_ = en_
                            # print('st_', st_)
                            ft = crd[:, p_idx, :]
                            # print(ft, ft.shape)
                            # ft = ft[1:, :]
                            ft = ft[:p]
                            # print(ft, ft.shape)
                            # print(pln, pln.shape)
                            t.append(self.mse(ft[:p], pln))
                        else : 
                            print('GOT ZERO POINT COMPONENET')
                        # print(t[-1])
                    t = torch.stack(t)
                    # print(t, t.shape) 
                    l3 = t.mean()
                    wt.add_scalar(spl +" Loss coordinate", l3.item(), (ep+1)*(ix+1))
                    print(spl +" Loss coordinate ", l3.item())
                else :
                    l2 = torch.zeros([], requires_grad=True)
                    l3 = torch.zeros([], requires_grad=True)
                print(spl +" Loss Decoder", l2.item())
                wt.add_scalar(spl +" Loss Decoder", l2.item(), (ep+1)*(ix+1))
                tr_l += l2
                tr_l += l3
               

            elif t == 'poolNet':
                op = x['op']
                pnl, pl, pt_idx = x['pnl']
                # print(pnl)
                # print('pn\npl')
                # print(pl)
                # print('pl\n inds')
                inds = point_to_index(pl)
                # print(inds)
                # print('inds\n idx')
                # print(pt_idx)
                hm_loss = []
                emb1_loss = []
                emb2_loss = []
                regr_loss = []
                for out in op :
                    hm, em, reg = out
                    # print('hm, em, reg')
                    # print(hm.shape, em.shape, reg.shape)
                    fm_mse = hm[:, 0:2, :, :]
                    fm_bce = hm[:, 2, :, :]
                    fm_ce  = torch.softmax(hm[:, 3:5, :, :],dim=1)
                    l1 = self.regr_loss(fm_mse, mse_lbl, self.mse_wt()) 
                    ### BCE
                    # print('lbl->',  lbl.shape, 'inp ->', (hm[0][:, 0, :, :].unsqueeze(dim=1)).shape)
                    l2 = self.rec_loss(fm_bce, bce_lbl, weight=self.bce_wt(bce_lbl))
                    ### CE
                    l3 = self.cls_loss(fm_ce, ce_lbl, weight=self.ce_wt()) 
                    hm_loss.append(self.alpha*l1+self.beta*l2 +self.gamma*l3)
                    wt.add_scalar(spl +" Loss MSE", l1.item(), (ep+1)*(ix+1))
                    wt.add_scalar(spl +" Loss BCE", l2.item(), (ep+1)*(ix+1))
                    wt.add_scalar(spl +" Loss CE", l3.item(), (ep+1)*(ix+1)) 
                    ############## 
                    pll , psh  = [], []  
                    coords = []
                    for ix in range(em.shape[0]) :
                        point_ae = _tranpose_and_gather_feature(em[ix, :, :, :], inds[ix]).squeeze(-1)
                        # print('point_ae', point_ae)
                        point_ae = point_ae.split(pt_idx[ix])
                        pus, pul = get_push_pull_lines(point_ae)
                        pll.append(pul)
                        psh.append(pus)
                        ##########
                        crd = _tranpose_and_gather_feature(reg[ix, :, :, :], inds[ix])
                        pnlix = torch.tensor(pnl[ix]).float().cuda()
                        coords.append(self.mse(crd, pnlix))
                        # print('crd', crd.shape)
                        # print('pnlix', pnlix.shape)
                        # print('mse', coords[-1])

                    # print('Final')
                    # print('pll', pll)
                    pll = sum(pll)/len(pll)
                    # print('pll', pll)
                    # print('psh', psh)
                    psh = sum(psh)/len(psh)
                    # print('psh', psh)
                    emb1_loss.append(psh)
                    emb2_loss.append(pll)
                    
                    # print('coords', coords)
                    coords = sum(coords)/len(coords)
                    regr_loss.append(coords)
                
                # print(hm_loss)
                # print(emb1_loss)
                # print(emb2_loss)
                # print(regr_loss)
                l0 = sum(hm_loss)
                l1 = sum(emb1_loss)
                l2 = sum(emb2_loss)
                l3 = sum(regr_loss)
                print(spl +" Loss hm ", l0.item())
                print(spl +" Loss push ", l1.item())
                print(spl +" Loss pull ", l2.item())
                print(spl +" Loss coordinate ", l3.item())
                wt.add_scalar(spl +" Loss Push", l1.item(), (ep+1)*(ix+1))
                wt.add_scalar(spl +" Loss Pull", l2.item(), (ep+1)*(ix+1))
                wt.add_scalar(spl +" Loss coordinate", l3.item(), (ep+1)*(ix+1))
                tr_l = (l0+l1+l2+l3)/len(op)
            elif t == 'poolfcnNet':
                hm, em, reg = x['op']
                pnl, pl, pt_idx = x['pnl']
                inds = point_to_index(pl)
                fm_mse = hm[:, 0:2, :, :]
                fm_bce = hm[:, 2, :, :]
                fm_ce  = torch.softmax(hm[:, 3:5, :, :],dim=1)
                l1 = self.regr_loss(fm_mse, mse_lbl, self.mse_wt()) 
                ### BCE
                l2 = self.rec_loss(fm_bce, bce_lbl, weight=self.bce_wt(bce_lbl))
                ### CE
                l3 = self.cls_loss(fm_ce, ce_lbl, weight=self.ce_wt()) 
                hm_loss = self.alpha*l1+self.beta*l2 +self.gamma*l3
                wt.add_scalar(spl +" Loss MSE", l1.item(), (ep+1)*(ix+1))
                wt.add_scalar(spl +" Loss BCE", l2.item(), (ep+1)*(ix+1))
                wt.add_scalar(spl +" Loss CE", l3.item(), (ep+1)*(ix+1)) 
                ############## 
                pll , psh  = [], []  
                coords = []
                for ix in range(em.shape[0]) :
                    point_ae = _tranpose_and_gather_feature(em[ix, :, :, :], inds[ix]).squeeze(-1)
                    # print('point_ae', point_ae)
                    point_ae = point_ae.split(pt_idx[ix])
                    pus, pul = get_push_pull_lines(point_ae)
                    pll.append(pul)
                    psh.append(pus)
                    ##########
                    crd = _tranpose_and_gather_feature(reg[ix, :, :, :], inds[ix])
                    pnlix = torch.tensor(pnl[ix]).float().cuda()
                    coords.append(self.mse(crd, pnlix))
                pll = sum(pll)/len(pll)
                # print('pll', pll)
                # print('psh', psh)
                psh = sum(psh)/len(psh)
                # print('psh', psh)
                emb1_loss = psh
                emb2_loss = pll
                # print('coords', coords)
                coords = sum(coords)/len(coords)
                regr_loss = coords
                
                l0 = hm_loss
                l1 = emb1_loss
                l2 = emb2_loss
                l3 = regr_loss
                print(spl +" Loss hm ", l0.item())
                print(spl +" Loss push ", l1.item())
                print(spl +" Loss pull ", l2.item())
                print(spl +" Loss coordinate ", l3.item())
                wt.add_scalar(spl +" Loss Push", l1.item(), (ep+1)*(ix+1))
                wt.add_scalar(spl +" Loss Pull", l2.item(), (ep+1)*(ix+1))
                wt.add_scalar(spl +" Loss coordinate", l3.item(), (ep+1)*(ix+1))
                # tr_l = (l0+l1+l2+l3)/len(op)
                tr_l = l0+l1+l2+l3
            else : 
                k = x['kp']
                r = x['rel']
                print('\n')
                training_loss = []
                for hm in k :
                    fm_mse = hm[:, 0:2, :, :]
                    fm_bce = hm[:, 2, :, :]
                    fm_ce  = torch.softmax(hm[:, 3:5, :, :],dim=1)
                    l1 = self.regr_loss(fm_mse, mse_lbl, self.mse_wt()) 
                    ### BCE
                    # print('lbl->',  lbl.shape, 'inp ->', (hm[0][:, 0, :, :].unsqueeze(dim=1)).shape)
                    l2 = self.rec_loss(fm_bce, bce_lbl, weight=self.bce_wt(bce_lbl))
                    ### CE
                    l3 = self.cls_loss(fm_ce, ce_lbl, weight=self.ce_wt()) 
                    training_loss.append(self.alpha*l1+self.beta*l2 +self.gamma*l3)
                    wt.add_scalar(spl +" Loss MSE", l1.item(), (ep+1)*(ix+1))
                    wt.add_scalar(spl +" Loss BCE", l2.item(), (ep+1)*(ix+1))
                    wt.add_scalar(spl +" Loss CE", l3.item(), (ep+1)*(ix+1))

                    # print(round(l1.item(), 5), round(l2.item(), 5), round(l3.item(), 5))
                
                l0 =  sum(training_loss)/len(x)

                # print('keypoint loss', l0)

                nodes, trg , edge_list= r 
                l4, l5 = self.node_pp_loss(nodes)
                wt.add_scalar(spl +" Loss node push", l5.item(), (ep+1)*(ix+1))
                wt.add_scalar(spl +" Loss node pull", l4.item(), (ep+1)*(ix+1))
                # print('In loss, nd', nodes[0].shape)
                # print('In loss, nd ct', nodes[1])
                # print('Total trg', len(trg))

                edg_cls = [_[1] for _ in edge_list]
                edg_emb = [_[0] for _ in edge_list]
                ec = torch.softmax(torch.cat(edg_cls), dim=-1)
                
                l2 = self.cls_loss(ec, torch.cat(trg).to(ec.device))
                wt.add_scalar(spl +" Loss Edge class", l2.item(), (ep+1)*(ix+1))
                # print('ed cls loss', l2)

                l1 = self.edge_pp_loss(edg_emb, trg)
                wt.add_scalar(spl +" Loss MS edg", l1.item(), (ep+1)*(ix+1))
                
                tr_l = 0.6*l0 + 0.1*l1 + 0.1*l2+ 0.1*l4 + 0.1*l5
                print('*'*80)
                print(l0.item(), l1.item(), l2.item(), l4.item(), l5.item())
                print(round(0.6*l0.item(), 5), round(0.1*l1.item(), 5), round(0.1*l2.item(), 5), round(0.1*l4.item(), 5), round(0.1*l5.item(), 5))
                print('*'*80)
                # exit()

        else : 
            print('NOT IMPLEMENTED')
            print(x.keys())
            # print(x.shape)
            exit()
        
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
            # print('-->>', loss.item())
            
        return loss / num_joints



class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha #(int exponent) ==2 
        self.beta = beta #(int exponent) == 4
    def forward(self, output, target, sigmoid=True):
        if sigmoid:  # clamp is important
            ## output is heatmap output in raw featuremap values
            ## applying sigmoid gives probablities 
            ## applying clamp ensures no 'zero-error' 
            output = torch.clamp(output.sigmoid(), min=1e-4, max=1 - 1e-4)
        pos_index = target.eq(1).float()
        ## pos_index gets all the positive indices from target heat map
        ## a.k.a. indices of cels which have value == 1  
        neg_index = target.lt(1).float()
        ## neg_index gets all the negative indices from target heat map
        ## a.k.a. indices of cels which have value < 1 
        pos_loss = torch.pow(1 - output, self.alpha) * torch.log(output) * pos_index
        ## pos_loss positive loss is pull loss (pulls similar ie positive sample tgthr)
        ## 1-output gives all inverse probablites, its squared and multiplied with log prob of prediction and masked by positive index   
        neg_loss = torch.pow(1 - target, self.beta) * torch.pow(output, self.alpha) * torch.log(1 - output) * neg_index
        ## neg_loss positive loss is push loss (pushes similar ie positive sample tgthr)
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        pos_num = pos_index.sum()
        loss = 0
        loss = loss - (pos_loss + neg_loss) / pos_num if pos_num > 0 else loss - neg_loss
        return loss

 

class PPLoss(nn.Module):
    def __init__(self):
        super(PPLoss, self).__init__() 
        self.thresh = 1e-6
        self.margin = 8
    
    def forward(self, nodes) :
        feats, cts, sub_comp_idx = nodes
        # feats = F.normalize(feats)
        # print('*'*40) #, feats)
        # print('feats in pploss', feats.shape ) #, feats)
        # print(len(cts), cts)
        pos_batch = []
        neg_batch = []
        pos = []
        neg = []
        # means = []
        for cix, c in enumerate(cts) : 
            # pos = []
            # neg = []
            start = sum(cts[:cix])
            # print('start, count', start, cix, c)
            f = feats[start:start+c]
            f = F.normalize(f)
            # print('feta selected ', f.shape)
            
            for lin in sub_comp_idx[cix]:
                tot = f.shape[0]
                ix_ = list(set([i__ for i__ in range(tot)]) - set(lin))
                # print('select', lin, 'rest', ix_)
                pull = f[lin]
                # print('pull', pull.shape)
                k = torch.mean(pull, dim=0)
                s_ = F.pairwise_distance(pull, k.repeat(pull.shape[0],1))
                s_ = torch.mean(s_)
                pos.append(s_)

                push = f[ix_]
                if push.shape[0] > 0 :
                    # print('mean, pull dim0', k.shape)               
                    r_ = F.pairwise_distance(push, k.repeat(push.shape[0], 1), p=1)
                    # print('dist rest vs mean', r_.shape)
                    # print(r_, self.margin-r_, F.relu(self.margin - r_))
                    r__ = torch.mean(F.relu(self.margin - r_))
                    if not torch.isnan(r__) :
                        neg.append(r__)
                # else : 
                    # print(r_)
                    # print(self.margin - r_)
                    # print(F.relu(self.margin - r_))
                # print('apeend rest, mean(r_)', r_.shape, r_)
        if len(pos) > 0 :                
            pos = torch.stack(pos)
        else : 
            print('0 pos')
            print('feats, cts, sub_comp_idx')
            print(feats.shape, cts, sub_comp_idx)
            pos = torch.zeros([], requires_grad=True)

        if len(neg) > 0 :                
            neg = torch.stack(neg)
        else : 
            print('0 neg')
            print('feats, cts, sub_comp_idx')
            print(feats.shape, cts, sub_comp_idx)
            neg = torch.zeros([], requires_grad=True)
        # print('Final', pos.shape, neg.shape)
        # print('Final', pos, neg)
        # print('Final', torch.mean(pos), torch.mean(neg))
        return torch.mean(pos), torch.mean(neg)
        # return sum(pos)/len(pos), sum(neg)/len(neg)




class MultiSimilarityLoss(nn.Module):
    def __init__(self):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2.0
        self.scale_neg = 40.0

    def forward(self, featss, labelss):
        # print('in ms loss', len(featss), len(labelss))
        bl = []
        for feats, labels in zip(featss, labelss) :
            feats = F.normalize(feats)
            labels = labels.to(feats.device)
            assert feats.size(0) == labels.size(0), \
                f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
            batch_size = feats.size(0)
            sim_mat = torch.matmul(feats, torch.t(feats))
            # sim_mat = F.linear(feats, torch.t(feats))

            # print('f', feats.shape)
            # print('l', labels.shape)
            # print('b', batch_size)
            # print('sm', sim_mat.shape)
            # print('sm', sim_mat)


            epsilon = 1e-12
            loss = list()
            ct = 0 
            ct2 = 0 

            for i in range(batch_size):
                pos_pair_ = sim_mat[i][labels == labels[i]]
                # print(':: ', pos_pair_.shape)
                # print(pos_pair_)
                pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
                # print(pos_pair_.shape)
                # print(pos_pair_)
                neg_pair_ = sim_mat[i][labels != labels[i]]

                if pos_pair_.shape[0] < 1 or neg_pair_.shape[0] < 1:
                    # print('-->', pos_pair_.shape, neg_pair_.shape)
                    # print(i, feats.shape, labels.shape)
                    # print(labels[i], torch.sum(labels == labels[i]))
                    # # print(labels)
                    # print(sim_mat)
                    # exit()
                    ct+=1
                    continue
                neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
                pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]


                if len(neg_pair) < 1 or len(pos_pair) < 1:
                    # print(len(neg_pair), len(pos_pair))
                    ct2+=1
                    continue

                # weighting step
                pos_loss = 1.0 / self.scale_pos * torch.log(
                    1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
                neg_loss = 1.0 / self.scale_neg * torch.log(
                    1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
                loss.append(pos_loss + neg_loss)

            if len(loss) == 0:
                print('len( MSloss)', len(loss),'/', batch_size, ct, ct2 )
                loss = torch.zeros([], requires_grad=True)
            else :
                loss = sum(loss) / batch_size
            bl.append(loss)
        # print(bl)
        # exit()
        return sum(bl)/len(bl)