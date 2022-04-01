'''
Transformation function to add/remove textboxes from charts and rand transform
'''

import torchvision.transforms as T
import numpy as np 
import PIL
from PIL import Image
import PIL
import PIL.Image as Image
import json
import math
import torch
import numpy as np
import pickle 
import os 
import os.path as osp

from scipy.interpolate import interp1d
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
### Line sequence transform 
'''
Takes image 
Finds random crop within plot area containing lines 
generates binary mask with interpolation of random crop 
return crop, mask, points 
handles, PMC, synth, figureSeer 
'''
class LineSeqTransform():
    def __init__(self, arg=512):
        self.re_size_dim = arg
        self.crop_size_dim = 128
        self.pp1 = T.Compose([T.ToTensor(),T.Normalize(mean=0.5, std=0.5)])
        self.rz = T.Resize((self.re_size_dim,self.re_size_dim))
    def __call__(self, **kwds):
        input_ = kwds['chart']
        js_obj = kwds['js_obj']
        if kwds['split'] == 'figureSeer' :
            ln_data = js_obj['data']['curves']
            ctp = 'figureSeer'
            tb = None
            lp = None
            a, b, c, d = js_obj['data']['plotpos']
            # print('a, b, c, d', a, b, c, d)
            x0, x1 = min(a, c), max(a, c)
            y0, y1 = min(b, d), max(b, d)
            ploth, plotw = y1-y0, x1-x0
            # input_.crop((x0, y0, x1, y1)).show()   
            # input_.crop((x0, y0, plotw, ploth)).show()   
            all_points, all_ipoints= self.get_fs_lines(ln_data)
        else :            
            ln_data = js_obj['task6']['output']['visual elements']['lines']
            ctp = js_obj['task6']['input']['task1_output']['chart_type'] 
            tb =  js_obj['task6']['input']['task2_output']['text_blocks']
            lp =  js_obj['task6']['input']['task5_output']['legend_pairs']
            ploth, plotw , x0, y0 = js_obj['task6']['input']['task4_output']['_plot_bb'].values()
            all_points, all_ipoints= self.get_lines(ln_data)        
        # print(all_points)
        # print(all_ipoints)
        # px, py = [_[0] for k in all_points for _ in k ], [_[1] for k in all_points for _ in k]
        px, py = [_[0] for k in all_ipoints for m in k for _ in m ], [_[1] for k in all_ipoints for m in k for _ in m ]
        # print('sorted(px)',sorted(px))
        # print('sorted(py)',sorted(py))
        # print('min(px) < x0', min(px), x0)
        # print('min(py) < y0', min(py), y0)
        # print('1', ploth, plotw , x0, y0)
        # input_.show()
        # input_.crop((x0, y0, plotw, ploth)).show()        
        if min(px) > x0 :
            x0 = min(px)
        if max(px) < x0 + plotw :
            plotw =  max(px) - x0
        if min(py) > y0 :
            y0 = min(py)
        if max(py) < y0 + ploth :
            ploth =  max(py) - y0
        # print('2, after reduc', ploth, plotw , x0, y0)
        # print('chart size', input_.size)
        # print('im_n.shape', im_n.shape)
        crpx, crpy = self.get_crp_coords(ploth, plotw , x0, y0, px, py)
        # print('crp ::', crpx-self.crop_size_dim/2, crpy-self.crop_size_dim/2, crpx+self.crop_size_dim/2, crpy+self.crop_size_dim/2)
        # print('crp ::', crpx, crpy, crpx+self.crop_size_dim, crpy+self.crop_size_dim)
        # plt.imshow(im_n[crpy : crpy+self.crop_size_dim, crpx : crpx+self.crop_size_dim])
        # plt.show()
        im = input_.crop((crpx-self.crop_size_dim/2, crpy-self.crop_size_dim/2, crpx+self.crop_size_dim/2, crpy+self.crop_size_dim/2))
        # im.show()   
        # px_left = [_ for _ in px if _ >= crpx-self.crop_size_dim/2 and _ <= crpx+self.crop_size_dim/2 ]
        # py_left = [_ for _ in py if _ >= crpy-self.crop_size_dim/2 and _ <= crpy+self.crop_size_dim/2 ]
        # print('sorted(px left)',sorted(px_left))
        # print('sorted(py left)',sorted(py_left)) 
        ipt_left =[]
        xcrp_min, xcrp_mx = crpx-self.crop_size_dim/2, crpx +self.crop_size_dim/2 
        ycrp_min, ycrp_mx = crpy-self.crop_size_dim/2, crpy +self.crop_size_dim/2 
        for ln in all_ipoints :
            ipt_ =[]
            for ipoltd in ln : 
                # print('len, 0, -1 :', len(ipoltd), ipoltd[0], ipoltd[-1])
                for pt in ipoltd :
                    if (pt[0]<= xcrp_mx) and (pt[0]>=xcrp_min) and (pt[1]<=ycrp_mx) and (pt[1]>=ycrp_min) :
                        # print(pt, xcrp_min, xcrp_mx, ycrp_min, ycrp_mx )
                        ipt_.append((pt[0]-xcrp_min, pt[1]-ycrp_min))
            if len(ipt_) > 0 :
                ipt_left.append(ipt_)
                px, py = [_[0] for _ in ipt_], [_[1] for _ in ipt_]
                # print('maxpx', max(px), max(py))
                # plt.scatter(px, py) 
                # plt.scatter(py, px) 
        # plt.show()
        cw, ch = im.size
        canvas = torch.zeros(cw, ch)
        im = self.rz(im)
        im = self.pp1(im)
        # print(im.shape)
        # if len(im.shape) == 3 :
            # im = im.unsqueeze(0)
        # print(im.shape)
        point_list = []
        for l_ in ipt_left :
            # print('\n', len(l_))
            pl = []
            for pts_ in l_ :
                # print(pts_)
                # print(int(pts_[1]-1), int(pts_[0]-1))
                pl.append((pts_[1]-1, pts_[0]-1))
                canvas[int(pts_[1]-1), int(pts_[0]-1)]+=1  
            point_list.append(pl)
        canvas = gaussian_filter(canvas, 1)
        # return im, ipt_left, torch.from_numpy(canvas)
        return im, point_list, torch.from_numpy(canvas)
    def get_crp_coords(self, ploth, plotw , x0, y0, px, py):
        selx = []
        for x_ in px : 
            # if x_ >= (self.crop_size_dim/2) + x0 and x_ +(self.crop_size_dim/2) <= plotw :  ## Inner plot crop 
            # print(x_, (self.crop_size_dim/2), x_+ (self.crop_size_dim/2), (plotw +x0))
            if x_ >= (self.crop_size_dim/2) and x_ +(self.crop_size_dim/2) <= (plotw +x0):
                selx.append(x_)
        # print(selx)
        selc_x = np.random.choice(selx)
        ix = px.index(selc_x)
        selc_y = py[ix]
        return selc_x, selc_y
    def get_lines(self, ln_data):
        pt = []
        ipt = []
        for ln in ln_data :
            ptl = [] 
            iptl = [] 
            for ix, pt_ in enumerate(ln) :
                px= pt_['x']; py =pt_['y']
                ptl.append(list((px, py)))
                if ix > 0 : 
                    ppx, ppy = ln[ix-1]['x'],  ln[ix-1]['y']
                    iptl.append(self.interpolt(ppx, ppy, px, py))
            pt.append(ptl)
            ipt.append(iptl)
        return pt, ipt
    def get_fs_lines(self, ln_data):
        pt = []
        ipt = []
        for ln in ln_data :
            iptl = [] 
            x = ln['x']
            y = ln['y']
            ptl = [[a[0], b[0]] for a, b in zip(x, y)] 
            pt.append(ptl)
        for p_ in pt :  
            for ix, pt_ in enumerate(p_) :
                px= pt_[0]; py =pt_[1]
                if ix > 0 : 
                    ppx, ppy = p_[ix-1][0],  p_[ix-1][1]
                    iptl.append(self.interpolt(ppx, ppy, px, py))
            ipt.append(iptl)
        return pt, ipt
    def interpolt(self, x1,y1, x2,y2, p=20) :
        # print(x1,y1, x2,y2)
        ps = []
        r = max(x2, x1) - min(x2, x1)
        if r != 0 : 
            # print('x', round(r,3), round(r/p, 3))
            z = interp1d([x1,x2],[y1, y2],fill_value="extrapolate")
            xs = np.arange(min(x2, x1), max(x2, x1), r/p)
            ys = z(xs)
            for ix, x in enumerate(xs):
                # ps.append((int(ys[ix]), int(x)))
                ps.append((int(x), int(ys[ix])))
        else :
            r = max(y2, y1) - min(y2, y1)
            if r != 0 : 
                # print('y', round(r, 3), round(r/p, 3))
                z = interp1d([y1, y2], [x1,x2], fill_value="extrapolate")
                ys = np.arange(min(y2, y1), max(y2, y1), r/p)
                xs = z(ys)
                for iy, y in enumerate(ys):
                    # ps.append((int(ys[ix]), int(x)))
                    ps.append((int(xs[iy]), int(y)))
        return ps     
        



### Chart Text box transform 
### Adds or removes text boxes with given random probablity
#   pp1 - tensor, normalize, 
#   pp2 - color jitter , tensor, norm. 
#   rz  - resize. 
### For Adobe+PMC -   probablity (text box, pp1, pp2) + rz 
### For figure seer   pp2 + rz 
### For test          pp1 + rz 
class ChartTransform():
    def __init__(self, arg=512):
        self.re_size_dim = arg
        self.pp1 = T.Compose([T.ToTensor(),T.Normalize(mean=0.5, std=0.5)])
        self.pp2 = T.Compose([T.ColorJitter(brightness=[0.5, 1], contrast=[0.2, 0.5], saturation=[0.2, 0.7], hue=[0, 0.4]), T.ToTensor(),T.Normalize(mean=0.5, std=0.5)])
        self.rz = T.Resize((self.re_size_dim,self.re_size_dim))
        
    def __call__(self, **kwds):
        input_ = kwds['chart']
        # print('ct::', input_.size)
        js_obj = kwds['js_obj']
        if kwds['split'] == 'test' :
            img_ = self.pp1(input_)
            return self.rz(img_)
        if kwds['split'] == 'figureSeer' :
            img_ = self.pp2(input_)
            return self.rz(img_)

        # hm = kwds['hm']
        chz = np.random.choice([0,1, 2, 3], 1, p=[0.3, 0.15, 0.15,  0.4])
        
        ctp = js_obj['task6']['input']['task1_output']['chart_type'] 
        tb =  js_obj['task6']['input']['task2_output']['text_blocks']
        # print('textbox found', len(tb))
        lp =  js_obj['task6']['input']['task5_output']['legend_pairs']
        # print('legendpair found', len(lp))
        img_ = np.array(input_)
        chz = chz[0]
        # print('Chz', chz)
        if chz == 0 or chz == 1:
            # print('remove textbx')
            for bx in tb :
                poly = bx['polygon'] if 'polygon' in bx else bx['bb']                
                # print(poly)
                x_min = min(int(poly['x0']), int(poly['x1']), int(poly['x2']), int(poly['x3']))
                x_max = max(int(poly['x0']), int(poly['x1']), int(poly['x2']), int(poly['x3']))
                y_min = min(int(poly['y0']), int(poly['y1']), int(poly['y2']), int(poly['y3']))
                y_max = max(int(poly['y0']), int(poly['y1']), int(poly['y2']), int(poly['y3']))
                # print(x_min,x_max, y_min,y_max)
                img_[y_min:y_max, x_min :x_max, :] = 255
        if chz == 0  or chz==2:
            # print('remove legend')
            for bx in lp : 
                x_min = bx['bb']['x0']
                y_min = bx['bb']['y0']
                x_max = bx['bb']['x0'] + bx['bb']['width']
                y_max = bx['bb']['y0']+ bx['bb']['height']
                img_[y_min:y_max, x_min :x_max, :] = 255            

        chz2 = np.random.choice([0,1], 1, p=[0.7, 0.3])
        # print(chz2)
        if chz2[0] == 1 :
            # print('image jitter')
            img_ = self.pp2(Image.fromarray(img_))
        else : 
            # print('image tensor')
            img_ = self.pp1(img_)
        # print('before rz', img_.shape)
        img_ = self.rz(img_)
        return img_  


import pickle 
import os 
import os.path as osp
import json
import PIL 
from PIL import Image
from torchvision.utils import save_image, make_grid
# # if __name__ == "__main__":

# # ln_lst = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/split4_train_task6_line.pkl'
# # with open(ln_lst, 'rb') as f :
# #     ln_lst = pickle.load(f)

# js_ = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/line'
# img_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/line'

# fl = PMC5457103___materials-09-00755-g007.json
# img  = PMC5457103___materials-09-00755-g007.jpg
# LST = LineSeqTransform()
# # LST(chart=im, js_obj=js_obj, split='Line')

# js_obj = json.load(open('/home/sahmed9/Documents/data/charts/FigureSeer/FigureSeerDataset/Annotations/01184-10.1.1.228.5357-Figure-7.json', 'r'))
# im = Image.open('/home/sahmed9/Documents/data/charts/FigureSeer/FigureSeerDataset/Annotated images/01184-10.1.1.228.5357-Figure-7.png')

# js_obj = json.load(open('/home/sahmed9/Documents/data/charts/FigureSeer/FigureSeerDataset/Annotations/01650-10.1.1.357.5457-Figure-8.json', 'r'))
# im = Image.open('/home/sahmed9/Documents/data/charts/FigureSeer/FigureSeerDataset/Annotated images/01650-10.1.1.357.5457-Figure-8.png')

# for lix, l in enumerate(ln_lst) :
#     js_obj = json.load(open(osp.join(js_, l)))
#     im = Image.open(osp.join(img_dir, l[:-4]+'jpg'))
#     im = im.convert('RGB')
#     # im.show()
#     # imd = ImageDraw.Draw(im)
#     z, p, c  = LST(chart=im, js_obj=js_obj, split='Line')
#     sc_dir = '/home/sahmed9/Documents/data/tash/tmp_'+str(lix)+'_.png'
#     save_image(make_grid([z, c.unsqueeze(0).repeat(3, 1, 1)]),sc_dir)
#     if lix > 50 :
#         break



# for i in range(50): 
#     z, p, c  = LST(chart=im, js_obj=js_obj, split='figureSeer')
#     sc_dir = '/home/sahmed9/Documents/data/tash/tmp_'+str(i)+'0_.png'
#     save_image(z , sc_dir) 
#     sc_dir = '/home/sahmed9/Documents/data/tash/tmp_'+str(i)+'1_.png'
#     save_image(c.unsqueeze(0).repeat(3, 1, 1),sc_dir) 
  

    


# for i in range(150): 
#     z, p, c  = LST(chart=im, js_obj=js_obj, split='Line')
#     sc_dir = '/home/sahmed9/Documents/data/tash/tmp_'+str(i)+'0_.png'
#     save_image(z , sc_dir) 
#     sc_dir = '/home/sahmed9/Documents/data/tash/tmp_'+str(i)+'1_.png'
#     save_image(c.unsqueeze(0).repeat(3, 1, 1),sc_dir) 

# if __name__ == '__main__' :
#     sv_img = '/home/sahmed9/reps/chart_add_style/output/line/img/'
#     sv_mask = '/home/sahmed9/reps/chart_add_style/output/line/mask/'
#     sv_ann = '/home/sahmed9/reps/chart_add_style/output/line/points/'
#     sv_plot = '/home/sahmed9/reps/chart_add_style/output/line/plot/'
#     grid = '/home/sahmed9/reps/chart_add_style/output/line/grid'
#     fl = '/home/sahmed9/reps/chart_add_style/split4_train_task6_line.pkl'
#     jsd = '/home/sahmed9/reps/KeyPointRelations/data/JSONs/line/'
#     imd = '/home/sahmed9/reps/KeyPointRelations/data/images/line'
#     op_plot_sz = 128
#     LST = LineSeqTransform(op_plot_sz)
#     f = pickle.load(open(fl, 'rb'))
#     for js_ix, js in enumerate(f['line']) : 
#         print('\n')
#         print('*'*80)
#         print(js)
#         print('*'*80)
#         js_obj = json.load(open(osp.join(jsd, js)))
#         img = Image.open(osp.join(imd, js[:-4]+'jpg'))
#         g_img = [Image.open(osp.join(grid, _)) for _ in os.listdir(grid)]
#         w, h = img.size
#         z, p, c  = LST(chart=img, js_obj=js_obj, split='Line')
#         sc_dir = osp.join(sv_mask, js[:-5]+'.jpg')
#         save_image(c.unsqueeze(0).repeat(3, 1, 1),sc_dir) 
#         sc_dir = osp.join(sv_img, js[:-5]+'.jpg')
#         save_image(z,sc_dir) 
#         sc_dir = osp.join(sv_ann, js[:-5]+'.npy')
#         # print('ipt_left', p)
#         p = np.array(p)
#         np.save(sc_dir, p)
        # print('np ->', p)
        # np.save(sc_dir, np.array([p, img.size], dtype=object))
        # if js_ix < 3 :
        #     for gi_ix, g in enumerate(g_img) : 
        #         g = g.resize((op_plot_sz,op_plot_sz) )
        #         plt.imshow(g)
        #         for p_ in p :
        #             print(p_)
        #             x = (p_[:, 0])
        #             y = (p_[:, 1])
        #             print('x', x)
        #             x/=w
        #             print('x/w', x)
        #             x*=128
        #             print('x128', x)
        #             y = (p_[:, 1])
        #             print('y', y)
        #             y/=h
        #             print('y/h', y)
        #             y*=128
        #             print('y128', y)
        #             y=max(y)-y
        #             plt.plot(x,y)
        #         sc_dir = osp.join(sv_plot, js[:-5]+str(gi_ix)+'_.jpg')
        #         plt.savefig(sc_dir)
        #         plt.clf()
        #     break
        # break

