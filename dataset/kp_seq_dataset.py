import torch
import PIL
import pickle 
import os 
import os.path as osp 
import json
from torch.utils.data import Dataset
from dataset.chart_transform import ChartTransform
from dataset.chart_transform import LineSeqTransform
from dataset.gt_json_transform import GroundTruth

class PMC_line(Dataset):
    def __init__(self, **kwd):
        
        config = kwd['config']
        self.config = config
        self.split = self.config.split
        self.im_dir = osp.join(config.rd, config.img_d)
        self.json_dir = osp.join(config.rd, config.js_d)
        self.json_test_dir = osp.join(config.rd, config.t_js_d)
        fl_list1 = osp.join(config.rd, config.fl_pmc)
        fl_list2 = osp.join(config.rd, config.fl_synth)
        fl_list3 = osp.join(config.rd, config.fl_fs)
        
        TRAIN_ALL = True
        self.crop = False
        self.map_sz = config.targ_sz
        self.ct = ChartTransform()
        self.gt = GroundTruth(mode='run', ch=None)
        ff1 = open(fl_list1, 'rb')
        ff1 = pickle.load(ff1)
        print('len fl_pmc', len(ff1))
        ff2 = open(fl_list2, 'rb')
        print('len fl_syn', len(ff2))
        ff2 = pickle.load(ff2)
        ff3 = open(fl_list3, 'rb')
        print('len fl_fs', len(ff3))
        ff3 = pickle.load(ff3)
        if  TRAIN_ALL :
            ff= [osp.join(folder, file) for folder in ff1 for file in ff1[folder] ]  
            print('loaded from ff1', len(ff1))
            ff+= [osp.join(folder, file) for folder in ff2 for file in ff2[folder] ]        
            print('loaded from ff2', len(ff2))
            ff+= [osp.join(folder, file) for folder in ff3 for file in ff2[folder] ]        
            print('loaded from ff3', len(ff3))
        
        splt_val = int(0.9 * len(ff))
      
        self.files={
            'train' : ff[:splt_val],
            'val' : ff[splt_val:],
            # 'test': os.listdir(self.json_test_dir)
            'test': ff
            }
        print('Loaded Total {} charts {} train, {} val from im dir {} and json dir {}'.format(len(ff), len(self.files['train']),len(self.files['val']), self.im_dir, self.json_dir))

    def __len__(self):
        return len(self.files[self.split])
    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        print(data_file)
        chart_type = data_file.split('/')[0]
        if self.split == 'test' :
            chart_type = 'line'
            chart_type = 'figureSeer'
            print('Found split TEST setting type to :', chart_type)
        img_file = osp.join(self.im_dir, data_file[:-4]+'jpg')
        if not osp.isfile(img_file) :
            img_file = osp.join(self.im_dir, data_file[:-4]+'png')
        if self.split=='test' :
            lbl_file = osp.join(self.json_test_dir, data_file) 
        else : 
            lbl_file = osp.join(self.json_dir, data_file)
        
        # load image
        img = PIL.Image.open(img_file)
        width, height = img.size
        js_obj= json.load(open(lbl_file,'r'))

        if self.crop :
            ploth, plotw , x0, y0 = js_obj['task6']['input']['task4_output']['_plot_bb'].values()
            img = img.crop((x0, y0, x0+plotw, y0+ploth))
        
        img = img.convert('RGB')  
        split_for_transform = self.split if chart_type != 'figureSeer' else 'figureSeer'
        img = self.ct(chart=img, js_obj=js_obj, split=split_for_transform)
        img = img.float()
        pt_       = [] 
        canvas    = []
        edge_list = []
        pt_norm   = []
        canvas = torch.zeros((self.map_sz, self.map_sz))
        canvas, edge_list, pt_, pt_norm = self.gt.generate_masks(chart_type, js_obj, canvas, width, height, data_file)
        if self.split != 'test' :
            img.requires_grad = True
            rt = {'image':img.float(), 'trg': canvas,  'el':edge_list, 'pt': pt_, 'pt_n':pt_norm, 'sz':[width, height]}
        else :
            rt =  {'image':img.float(), 'trg': data_file, 'el':edge_list, 'pt': pt_, 'pt_n':pt_norm, 'sz':[width, height]}
        return rt
    @staticmethod
    def collate_fn(batch):
        targets = []
        imgs = []
        edge_list = []
        g_pt = []
        g_pt_idx = []
        g_pt_n = []
        sz = []
        stck_trg = True
        for sample in batch:
            imgs.append(sample["image"])
            sz.append(sample["sz"])
            if isinstance(sample["trg"], str) :
                stck_trg = False
                targets.append(sample["trg"])
            else :
                targets.append(torch.from_numpy(sample["trg"]))
            edge_list.append(sample["el"])
            _ = [tuple(a_) for a__ in sample["pt"] for a_ in a__]
            g_pt.append(_)
            
            ky = []
            line_size = []
            for a_ln in sample["pt"] : 
                k = []
                print('len(each line)', len(a_ln))
                line_size.append(len(a_ln))
                for a_pt in a_ln :
                    k.append(list(_).index(tuple(a_pt)))
                ky.append(k)
            g_pt_idx.append(line_size)
            _ = [tuple(_) for _ in sample["pt_n"]]
            g_pt_n.append(_)
        if stck_trg :
            targets = torch.stack(targets, 0)
        return torch.stack(imgs, 0), targets,  edge_list, (g_pt, g_pt_idx), g_pt_n, sz
