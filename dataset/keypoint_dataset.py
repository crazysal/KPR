import torch
import PIL
import pickle 
import os 
import os.path as osp 
import json
from torch.utils.data import Dataset
from dataset.chart_transform import ChartTransform, LineSeqTransform
from dataset.gt_json_transform import GroundTruth
import numpy as np 
class PMC_line(Dataset):
    def __init__(self, **kwd):
        # split='train', loss='BCE', 
        config = kwd['config']
        self.config = config
        self.split = self.config.split
        self.im_dir = osp.join(config.rd, config.img_d)
        self.json_dir = osp.join(config.rd, config.js_d)
        self.json_test_dir = osp.join(config.rd, config.t_js_d)
        fl_list1 = osp.join(config.rd, config.fl_pmc)
        fl_list2 = osp.join(config.rd, config.fl_synth)
        fl_list3 = osp.join(config.rd, config.fl_fs)
        # print(config.rd, config.fl_pmc)
        # print(osp.join(config.rd, config.fl_pmc))
        # print(fl_list1)
        #allPMC U Synth

        TRAIN_ALL = config.split
        self.crop = False
        self.map_sz = config.targ_sz
        self.CT = ChartTransform()
        self.LST = LineSeqTransform()
        self.GT = GroundTruth(mode='run', ch=None)
        ff1 = open(fl_list1, 'rb')
        ff1 = pickle.load(ff1)
        # print('len fl_pmc', len(ff1.items()))
        # op_dir = '/home/sahmed9/reps/KeyPointRelations/cache/hms/9_clean_train/figureSeer/'
        # op_dir = os.listdir(op_dir)
        # op_dir = [_[:-4] for _ in op_dir]
        ff2 = open(fl_list2, 'rb')
        ff2 = pickle.load(ff2)
        ff3 = open(fl_list3, 'rb')
        ff3 = pickle.load(ff3)
        if  TRAIN_ALL !='test':
            ff= [osp.join(folder, file) for folder in ff1 for file in ff1[folder] ]  
            print('from ff1', len(ff))
            ff+= [osp.join(folder, file) for folder in ff2 for file in ff2[folder] ]        
            print('from ff2', len(ff))
            ff+= [osp.join(folder, file) for folder in ff3 for file in ff3[folder] ]        
            print('from ff3', len(ff))
            # ff= [osp.join(folder, file) for folder in ff1 for file in ff1[folder] if file[:-5] not in op_dir]        
        else :
            ff = os.listdir(self.json_test_dir)
            clean = ['PMC4132901___12889_2014_6915_Fig1_HTML.json', 'PMC4322732___fgene-06-00038-g002.json']
            ff = [_ for _ in ff if _ not in clean]
            # not in clean 
            # PMC4132901___12889_2014_6915_Fig1_HTML.json
            # PMC4322732___fgene-06-00038-g002.json
        splt_val = int(0.9 * len(ff))
        # print(ff)
        # print(op_dir)
        # exit()
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
            chart_type = 'figureSeer'
            chart_type = 'line'
            print('Found split TEST setting type to :', chart_type)
        img_file = osp.join(self.im_dir, data_file[:-4]+'jpg')
        if not osp.isfile(img_file) :
            img_file = osp.join(self.im_dir, data_file[:-4]+'png')
        if self.split=='test' :
            lbl_file = osp.join(self.json_test_dir, data_file) 
        else : 
            lbl_file = osp.join(self.json_dir, data_file)
        print('json from', lbl_file)
        
        # load image
        img = PIL.Image.open(img_file)
        width, height = img.size
        js_obj= json.load(open(lbl_file,'r'))

        if self.crop :
            ploth, plotw , x0, y0 = js_obj['task6']['input']['task4_output']['_plot_bb'].values()
            img = img.crop((x0, y0, x0+plotw, y0+ploth))
        
        img = img.convert('RGB')  
        split_for_transform = self.split if chart_type != 'figureSeer' else 'figureSeer'
        
        
        pt_       = [] 
        canvas    = []
        edge_list = []
        pt_norm   = []
        # canvas = torch.zeros((self.map_sz, self.map_sz))
        # canvas, edge_list, pt_, pt_norm = self.GT.generate_masks(chart_type, js_obj, canvas, width, height, data_file)
        if self.split != 'test' :
            img, pt_, canvas = self.LST(chart=img, js_obj=js_obj, split=split_for_transform)
            img.requires_grad = True
            # print('before collate', img.shape,  canvas.shape, len(edge_list), len(pt_), len(pt_norm))
            # print('#kp in each line')
            # for p in pt_ : 
            #     print(len(p))
            rt = {'image':img.float(), 'trg': canvas,  'el':edge_list, 'pt': pt_, 'pt_n':pt_norm, 'sz':[width, height]}
        else :
            img = self.CT(chart=img, js_obj=js_obj, split='Test')
            # print('-...............', img, img_file )   
            # return {'image':img.float(), 'trg': data_file,  'el':edge_list, 'pt': pt_, 'pt_n':pt_norm}  
            # g_pt = []
            # g_pt_idx = []
            # g_pt_n = []
            # _ = set([tuple(a_) for a__ in pt_ for a_ in a__])
            # g_pt.append(_)
            
            # ky = []
            # for a_ln in pt_ : 
            #     k = []
            #     for a_pt in a_ln :
            #         k.append(list(_).index(tuple(a_pt)))
            #     ky.append(k)
            # g_pt_idx.append(ky)
            # _ = set([tuple(_) for _ in pt_norm])
            # g_pt_n.append(_)
            # rt =  {'image':img.float(), 'trg': data_file,  'el':[edge_list], 'pt': (g_pt, g_pt_idx), 'pt_n':g_pt_n}  
            rt =  {'image':img.float(), 'trg': data_file, 'el':edge_list, 'pt': pt_, 'pt_n':pt_norm, 'sz':[width, height]}
        # print(rt)
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
            # print('-----------------------------', sample['sz'])
            imgs.append(sample["image"])
            sz.append(sample["sz"])
            # print('tt', type(sample["trg"]))
            if isinstance(sample["trg"], str) :
                stck_trg = False
                targets.append(sample["trg"])
            elif isinstance(sample["trg"], (np.ndarray, np.generic)) :
                targets.append(torch.from_numpy(sample["trg"]))
            else :
                targets.append(sample["trg"])
            # print('coll sample["el"]', len(sample["el"]))
            edge_list.append(sample["el"])
            # print(sample["pt"])
            # _ = set([tuple(_) for _ in sample["pt"]])
            # _ = set([tuple(a_) for a__ in sample["pt"] for a_ in a__])
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
            # g_pt_idx.append(ky)
            g_pt_idx.append(line_size)
            # _ = set([tuple(_) for _ in sample["pt_n"]])
            _ = [tuple(_) for _ in sample["pt_n"]]
            g_pt_n.append(_)
        # print('coll : edge_list["el"]', len(edge_list))
        if stck_trg :
            targets = torch.stack(targets, 0)
        return torch.stack(imgs, 0), targets,  edge_list, (g_pt, g_pt_idx), g_pt_n, sz




# class Seq_line(Dataset):
#     def __init__(self, **kwd):
#         # split='train', loss='BCE', 
#         config = kwd['config']
#         self.config = config
#         self.split = self.config.split
#         self.im_dir = osp.join(config.rd, config.img_d)
#         self.json_dir = osp.join(config.rd, config.js_d)
#         self.json_test_dir = osp.join(config.rd, config.t_js_d)
#         fl_list1 = osp.join(config.rd, config.fl_pmc)
#         fl_list2 = osp.join(config.rd, config.fl_synth)
#         fl_list3 = osp.join(config.rd, config.fl_fs)