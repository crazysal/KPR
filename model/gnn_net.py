import dgl
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from dgl.nn import GATConv
from dgl.nn import GlobalAttentionPooling
import copy 
class KeyPointRelationNet(nn.Module):
    ''''''
    def __init__(self, **kwd):
        super(KeyPointRelationNet, self).__init__()
        config = kwd['config']
        # chnl_ = hm1.shape[1]*3
        chnl_ = config.heatmap_chan = 256
        chnl_ *= config.stacks
        # self.m1 = nn.Conv2d(chnl_, 1, 1)
        # self.m2 = nn.Conv2d(chnl_, 1, 1)
        # self.m3 = nn.Conv2d(chnl_, 1, 1)
        self.m4 = nn.Conv2d(chnl_, config.heatmap_chan, 1)
        self.node_emb = nn.Linear(258, 128)
        
        self.gc1 = GATConv(128, 256, feat_drop=0.2, num_heads=3, allow_zero_in_degree=False)
        self.gate_nn1 = nn.Linear(256*3, 256)  
        
        self.gc2 = GATConv(256, 256, num_heads=3, allow_zero_in_degree=True)
        self.gate_nn2 = nn.Linear(256*3, 256) 
        
        self.gc3 = GATConv(256, 128, feat_drop=0.2, num_heads=3, allow_zero_in_degree=True)
        self.gate_nn3 = nn.Linear(128*3, 128) 
        
        self.edgeLin = nn.Linear(256, 2)

        self.config = config

    @staticmethod
    ### create list of batch dicts 
    ### each element in list is dict of points in one line
    ### node_dict -> {(kp_x, kp_y) : node_id}
    ### node id is incremental 0 - k  for k points in a line
    def get_batch_rev_dict(p_l):
        batch_dict_list = []
        # print('pl', p_l)
        for bt in p_l :
            i = 0
            node_dict = {}
            for p in bt:  
                if tuple(p) not in node_dict : 
                    node_dict.update({tuple(p):i})
                    i+=1
            rev_node_dict = dict((v,k) for k,v in node_dict.items())
            # print(len(node_dict))
            batch_dict_list.append(node_dict)
            # print(rev_node_dict)
            # for k in rev_node_dict : 
            #     batch_dict.update({rev_node_dict[k]:k})
        # print(batch_dict)
        # print(len(batch_dict), batch_dict.keys())
        # print(len(batch_dict_list))
        return batch_dict_list 




    def forward(self, x, edge_list, point_list_, point_norm_list):
        point_list, pt_idx = point_list_ 
        # point_list = [_ for a in point_list_ for _ in a]
        hm1, hm2, hm3 = x
        batch_dict_list = self.get_batch_rev_dict(point_list)
        # print('input to KRP')
        # print(hm1.shape)
        # print(hm2.shape)
        # print(hm3.shape)
        # print(pt_idx)
        # print(point_list)
        # print('Total pt list', len(point_list),len(point_norm_list))
        # s = 0
        # for p in point_list:
            # s+=len(p)
            # print('p in pt_list', len(p))
        # print(point_norm_list)
        # print('Total pt', s)
        maps = []
        for hix in range(hm1.shape[0]) : 
            maps.append(hm1[hix])
            maps.append(hm2[hix])
            maps.append(hm3[hix])
        maps = torch.stack(maps)
        maps = maps.reshape(int(maps.shape[0]/3), -1, maps.shape[-2], maps.shape[-1])
        # maps1 = maps.clone()
        # maps2 = maps.clone()
        # maps3 = maps.clone()
        # m1_ = self.m1(maps1)
        # m2_ = self.m2(maps2)
        # m3_ = self.m3(maps3)
        # maps_alpha = m1_*m2_
        # maps_alpha = torch.softmax(maps_alpha, dim=1)
        # maps_beta = (maps_alpha * m3_)
        # maps_beta = torch.sigmoid(maps_beta)
        # maps *= maps_beta
        maps = self.m4(maps)
        # print('After m4', maps.shape)
        nodes = []
        for graphs in range(maps.shape[0]) :
            # print('\n', graphs,'in', range(maps.shape[0]))
            op = []
            pl = point_list[graphs]
            # print(pl)
            pnl = point_norm_list[graphs]
            # print(pnl)
            # print('pts in order')
            ze = 0
            for pt_, ptn_  in zip(pl, pnl) :
                # print(ze, pt_[0], pt_[1])
                ze+=1
                fp = maps[graphs, :,  pt_[0], pt_[1]]
                # print(fp.shape)
                fp_ = fp.new([ptn_[0], ptn_[1]])
                # print(fp_.shape)
                raw = torch.cat((fp, fp_))
                # print(raw.shape)
                n_e = self.node_emb(raw)
                # print(n_e.shape)
                op.append(n_e)                  
            nodes.append(torch.stack(op))
        g_l = []
        tgaret = []
        ndoe_order = []
        for eix, e in enumerate(edge_list): 
            g_, t, n_order=  self.get_graph(e, batch_dict_list[eix])
            g_l.append(g_)
            tgaret.append(t)
            ndoe_order.append(n_order)
            # print(len(e))
            # print(nodes[eix].shape)
        
        # print('lenth tgaret', len(tgaret))
        # print('lenth ndoe_order', len(ndoe_order))

        node_ft = torch.cat(nodes, dim=0)
        g_l = dgl.batch(g_l).to(node_ft.get_device())

        # print('\nBatched', g_l)
        # print('node ft 1', node_ft.shape)
        node_ft = self.gc1(g_l, node_ft).reshape(node_ft.shape[0], -1)
        # print('node ft 2', node_ft.shape)       
        node_ft = F.leaky_relu(self.gate_nn1(node_ft))
        # print('node ft 3', node_ft.shape)       
        node_ft = self.gc2(g_l, node_ft).reshape(node_ft.shape[0], -1)
        # print('node ft 4', node_ft.shape)       
        node_ft = F.leaky_relu(self.gate_nn2(node_ft))
        # print('node ft 5', node_ft.shape)       
        node_ft = self.gc3(g_l, node_ft).reshape(node_ft.shape[0], -1)
        # print('node ft 6', node_ft.shape)       
        node_ft = F.leaky_relu(self.gate_nn3(node_ft))
        # print('node ft 7', node_ft.shape)    
        g_l = dgl.unbatch(g_l)
        n_nd = []
        for g__ in g_l : 
            n_nd.append(g__.num_nodes())

        edges_ = []    
        ed_ct = []
        for eix, e in enumerate(edge_list): 
            start_node = 0
            end_node = len(batch_dict_list[eix])
            edge_stack=  self.get_edge(e, batch_dict_list[eix], node_ft[start_node:end_node, :], ndoe_order[eix])
            edges_.append(edge_stack)
        
        # print('tolta batche edges', len(edges_))
        return ((node_ft, n_nd, pt_idx), tgaret, edges_)

    @staticmethod
    ### Generate graph per line
    ### u - source, v - sink 
    ### edge list format - [ [(kp_x1, kpy1) , edge_label, (kp_x2, kpy2)], [] .. ]
    ### n_order - keep track of edge order
    def get_graph(el_, batch_dict):
        trg = []
        edge_list = el_
        u = [batch_dict[tuple(_[0])] for _ in edge_list]
        trg = [_[1] for _ in edge_list]
        v = [batch_dict[tuple(_[2])] for _ in edge_list]
        n_order = [(batch_dict[tuple(_[0])], batch_dict[tuple(_[2])]) for _ in edge_list]
        g = dgl.graph((u, v))
        g = g.add_self_loop()
        # print(g, sum(g.in_degrees()), len(g.in_degrees()))
        # print('trg', len(trg), trg)
        # print('n_order', len(n_order), n_order)
        return g, torch.tensor(trg).long(), n_order
        
        
        
    def get_edge(self, el_, batch_dict, nodes, ndo_order):
        embs = []
        emb_cls = []
        # print('\n edge list')
        # print(el_)
        # print('\n batch dict')
        # print(batch_dict)
        for _ in ndo_order :
            i, j = _
            # print('i, j', _)
            e = [nodes[i], nodes[j]]
            # print(e[0].shape, e[1].shape)
            e = torch.cat(e)
            embs.append(e)
            # print(e.shape)
            e = self.edgeLin(e)
            emb_cls.append(e) 
            
            # print(e.shape)
        # return (embs,torch.stack(emb_cls))
        # print(' \ \ \ ')
        # print(len(embs))
        return (torch.stack(embs),torch.stack(emb_cls))

        
        
        




def krp(**kwd):
    model = KeyPointRelationNet(config=kwd['config'])
    return model