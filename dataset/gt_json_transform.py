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
class GroundTruth():
    def __init__(self, **kwargs):
        self.mode = kwargs['mode']
        self.choice = kwargs['ch']
        if self.choice == 2 :
            # Synth 
            self.img_dir = '/home/sahmed9/Documents/data/charts/ICPR_ChartCompetition2020_AdobeData/Chart-Images-and-Metadata/ICPR/Charts/'
            self.json_dir = '/home/sahmed9/Documents/data/charts/ICPR_ChartCompetition2020_AdobeData/Task-level-JSONs/JSONs/'
            fl_list = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/train_task6_syntyPMC.pkl'
            self.sd = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/synth_5_mask_interpolate_128/'
        elif self.choice == 3 :
            #PMC ALL
            self.img_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/'
            self.json_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON'
            fl_list = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/split4_train_task6_all.pkl'
            self.sd = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/all_pmc_mask_interpolate_128/'
        if self.mode =='save' :
            os.mkdir(self.sd) if not osp.isdir(self.sd) else print(self.sd, 'created')
            folder_list = ['vertical_box', 'vertical_bar', 'horizontal_bar','hbox','hGroup','hStack','line','scatter','line2','scatter2', 'vbox','vGroup','vStack']
            ff = pickle.load(open(fl_list, 'rb'))
            self.ff = [osp.join(folder, file) for folder in ff for file in ff[folder] if folder in folder_list]
    
        self.map_sz = 128
    
    @staticmethod
    ## Calculates gaussian 
    # for point (px128, py128)
    # plots on a as value sum for all pixels in range sz x sz
    def gauss(a, px128, py128, sz):
        for x in range(sz) :
            for y in range(sz) :
                vala = round(math.exp(-((x-px128)**2 + (y-py128)**2))/4, 1)
                if vala > 0 :
                    a[y, x] += vala    
        return a

    @staticmethod
    ## Interpolates p points bw (x1, y1) and (x2, y2), plots as pixel =1 on a
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
    def __call__(self, *args):
        for ix_, f in enumerate(self.ff) : 
            print('\n :::', ix_,'::: running for ', f)
            if self.choice == 1 or self.choice == 3 :
                img_file = osp.join(self.img_dir, f[:-4]+'jpg')
            else : 
                img_file = osp.join(self.img_dir, f[:-4]+'png')
            img = Image.open(img_file)
            width, height = img.size
            lbl_file = open(osp.join(self.json_dir, f), 'r')
            js_obj = json.load(lbl_file)
            print('wh', width, height)
            chart_type = f.split('/')[0]
            print('chart_type', chart_type)
            canvas = torch.zeros((self.map_sz, self.map_sz))
            self.generate_masks(chart_type, js_obj, canvas, width, height, f)
                

    @staticmethod
    def get_edge_list(pt, pnts):
        edge_list = []
        pt = [_ for a in pt for _ in a]
        points = np.array(pt)
        if len(set(points[:, 0])) == 1 or len(set(points[:, 1])) == 1 or len(pt) < 4 :
            for pix, p in enumerate(points):
                if pix > 0 : 
                    edge_list.append([list(points[pix-1]), 1, list(points[pix])])
            return edge_list
        tri = Delaunay(points)
        tring = points[tri.simplices]
        for t in tring : 
            p1, p2, p3 = t
            s1, s2, s3 = set(pnts[tuple(p1)]), set(pnts[tuple(p2)]), set(pnts[tuple(p3)])
            e1, e2, e3 = s1.intersection(s2), s1.intersection(s3), s2.intersection(s3)
            if len(e1) == 0 :
                edge_list.append([list(p1), 0, list(p2)])
            else:
                edge_list.append([list(p1), 1, list(p2)])
            if len(e2) == 0 :
                edge_list.append([list(p1), 0, list(p3)])
            else:
                edge_list.append([list(p1), 1, list(p3)])
            if len(e3) == 0 :
                edge_list.append([list(p2), 0, list(p3)])
            else:
                edge_list.append([list(p2), 1, list(p3)])
        return edge_list


    def process_lines(self, js_obj, canvas, width, height, f):
        ln_data = js_obj['task6']['output']['visual elements']['lines']
        map_sz = self.map_sz
        ptss = []
        pt_norm = []
        pnts = {}
        for ix, ln in enumerate(ln_data) : 
            pt = []
            for pt_ in ln : 
                xs = int((pt_['x'])*self.map_sz/width)
                ys = int((pt_['y'])*self.map_sz/height) 
                xs_norm = pt_['x']/width
                ys_norm = pt_['y']/height
                if xs == self.map_sz :
                    xs-=1
                if ys == self.map_sz :
                    ys-=1
                if xs < self.map_sz and ys < self.map_sz :
                    canvas[ys,xs] += 1#len(ln_data)
                    _ = (xs, ys)
                    pt.append(list(_))
                    pt_norm.append(list((xs_norm, ys_norm)))
                    if _ in pnts : 
                        pnts[_].append(ix)
                    else : 
                        pnts[_]=[ix]
                else: 
                    print('ERROR', xs, ys, self.map_sz, f)
                    # print(lod)
            ptss.append(pt)
        for ln in ln_data : 
            for ix, pt_ in enumerate(ln) :
                px128 = pt_['x']*map_sz/width; py128 =pt_['y'] *map_sz/height
                if ix >0 :
                    ppx128 = ln[ix-1]['x']*map_sz/width; ppy128 =ln[ix-1]['y'] *map_sz/height
                    canvas = self.interpolt(canvas, ppx128, ppy128, px128, py128)                
        canvas = gaussian_filter(canvas, 1)
        return canvas, self.get_edge_list(ptss, pnts), ptss, pt_norm,

    def process_boxplots(self, js_obj, canvas, width, height, f):
        map_sz = self.map_sz
        box_data = js_obj['task6']['output']['visual elements']['boxplots']
        ptss = []
        pt_norm = []
        pnts = {}
        for ix, box in enumerate(box_data) : 
            pt = []
            for pt_ in  box: 
                xs = int((box[pt_]['x'])*map_sz/width)
                ys = int((box[pt_]['y'])*map_sz/height)
                xs_norm = box[pt_]['x']/width
                ys_norm = box[pt_]['y']/height
                if xs == map_sz :
                    xs-=1
                if ys == map_sz :
                    ys-=1
                if xs < map_sz and ys < map_sz and xs >= 0 and ys>= 0 :
                    canvas[ys,xs] += 1#len(box_data)
                    _ = (xs, ys)
                    pt.append([xs, ys])
                    pt_norm.append(list((xs_norm, ys_norm)))
                    if _ in pnts : 
                        pnts[_].append(ix)
                    else : 
                        pnts[_]=[ix]
                else: 
                    print('ERROR', xs, ys, map_sz, f)   
                    # print(lod)
            ptss.append(pt)
        for ix, box in enumerate(box_data) : 
            mx = box['max']
            fq = box['first_quartile']
            med = box['median']
            tq = box['third_quartile']
            mn = box['min']
            interp_on = []
            # Top hr line 
            interp_on.append([(mx['_bb']['x0'], mx['_bb']['y0']), (mx['_bb']['x0'] + mx['_bb']['width'], mx['_bb']['y0'])])
            # Top vrt line 
            interp_on.append([(mx['x'], mx['y']), (fq['x'], fq['y'])])
            # FQ hr line 
            interp_on.append([(fq['_bb']['x0'], fq['_bb']['y0']), (fq['_bb']['x0'] + fq['_bb']['width'], fq['_bb']['y0'])])
            # FQ verr line left
            interp_on.append([(fq['_bb']['x0'], fq['_bb']['y0']), (med['_bb']['x0'], med['_bb']['y0'])])
            # FQ verr line right
            interp_on.append([(fq['_bb']['x0']+ fq['_bb']['width'], fq['_bb']['y0']), (med['_bb']['x0']+ med['_bb']['width'], med['_bb']['y0'])])
            # med hr line 
            interp_on.append([(med['_bb']['x0'], med['_bb']['y0']), (med['_bb']['x0'] + med['_bb']['width'], med['_bb']['y0'])])
            # med ver line left
            interp_on.append([(med['_bb']['x0'], med['_bb']['y0']), (tq['_bb']['x0'] , tq['_bb']['y0'])])
            # med ver line right
            interp_on.append([(med['_bb']['x0'] + med['_bb']['width'], med['_bb']['y0']), (tq['_bb']['x0'] + tq['_bb']['width'], tq['_bb']['y0'])])
            # tq hr line 
            interp_on.append([(tq['_bb']['x0'], tq['_bb']['y0']), (tq['_bb']['x0'] + tq['_bb']['width'], tq['_bb']['y0'])])
            # Lowe vewrt line 
            interp_on.append([(tq['x'], tq['y']), (mn['x'], mn['y'])])
            # mn hr line 
            interp_on.append([(mn['_bb']['x0'], mn['_bb']['y0']), (mn['_bb']['x0'] + mn['_bb']['width'], mn['_bb']['y0'])])
            for pts_ in interp_on :
                (i1, i2), (i3, i4) = pts_
                x1 = i1*map_sz/width
                y1 = i2*map_sz/height
                x2 = i3*map_sz/width
                y2 = i4*map_sz/height
                canvas = self.interpolt(canvas, x1, y1, x2, y2, 10)
                    
            # canvas = self.gauss(canvas, mx['x'], mx['y'], map_sz)
            # canvas = self.gauss(canvas, fq['x'], fq['y'], map_sz)
            # canvas = self.gauss(canvas, med['x'], med['y'], map_sz)
            # canvas = self.gauss(canvas, tq['x'], tq['y'], map_sz)
            # canvas = self.gauss(canvas, mn['x'], mn['y'], map_sz)
        canvas = gaussian_filter(canvas, 1)
        return canvas, self.get_edge_list(ptss, pnts), ptss, pt_norm
         
    def process_scatter_synth(self, js_obj, canvas, width, height, f):
        pt_data = js_obj['task6']['output']['visual elements']['scatter points']
        # print(len(pt_data))
        map_sz = self.map_sz
        ptss = []
        pt_norm = []
        pnts = {}
        pt_datas = [pt_data]
        # for pts in pt_datas : 
        #     for pt_ in pts: 
        #         px128 = pt_['x']*map_sz/width; py128 = pt_['y']*map_sz/height
        #         canvas = self.gauss(canvas, px128, py128, map_sz)
        for ix, pts in enumerate(pt_datas) : 
            pt = []
            for pt_ in pts: 
                px128 = int((pt_['x'])*map_sz/width)
                py128 = int((pt_['y'])*map_sz/height)
                xs_norm = pt_['x']/width
                ys_norm = pt_['y']/height
                if px128 == self.map_sz :
                    px128-=1
                if py128 == self.map_sz :
                    py128-=1
                if px128 < map_sz and py128 < map_sz and px128 >= 0 and py128>= 0 :
                    canvas[py128,px128] += 1#len(pt_data)
                    _ = (px128, py128)
                    pt.append([px128, py128])
                    pt_norm.append(list((xs_norm, ys_norm)))
                    if _ in pnts : 
                        pnts[_].append(ix)
                    else : 
                        pnts[_]=[ix]
                else: 
                    print('ERROR', px128, py128, map_sz, f)   
                    print(lod)
            ptss.append(pt)
        canvas = gaussian_filter(canvas, 1)
        return canvas, self.get_edge_list(ptss, pnts), ptss, pt_norm

    def process_scatter_pmc(self, js_obj, canvas, width, height, f):
        pt_data = js_obj['task6']['output']['visual elements']['scatter points']
        map_sz = self.map_sz
        ptss = []
        pt_norm = []
        pnts = {}
        # for pts in pt_data : 
        #     for pt_ in pts: 
        #         px128 = pt_['x']*map_sz/width; py128 = pt_['y']*map_sz/height
        #         canvas = self.gauss(canvas, px128, py128, map_sz)
        for ix, pts in enumerate(pt_data) : 
            pt = []
            for pt_ in pts: 
                px128 = int((pt_['x'])*map_sz/width)
                py128 = int((pt_['y'])*map_sz/height)
                xs_norm = pt_['x']/width
                ys_norm = pt_['y']/height
                if px128 == self.map_sz :
                    px128-=1
                if py128 == self.map_sz :
                    py128-=1
                if px128 < map_sz and py128 < map_sz and px128 >= 0 and py128>= 0 :
                    canvas[py128,px128] += 1#len(pt_data)
                    _ = (px128, py128)
                    pt.append([px128, py128])
                    pt_norm.append(list((xs_norm, ys_norm)))
                    if _ in pnts : 
                        pnts[_].append(ix)
                    else : 
                        pnts[_]=[ix]
                else: 
                    print('ERROR', px128, py128, map_sz, f)   
                    print(lod)
            ptss.append(pt)
        canvas = gaussian_filter(canvas, 1)
        return canvas, self.get_edge_list(ptss, pnts), ptss, pt_norm
                
    def process_h_bar(self, js_obj, canvas, width, height, f):
        bar_data = js_obj['task6']['output']['visual elements']['bars']
        map_sz = self.map_sz
        ptss = []
        pt_norm = []
        pnts = {}

        for ix, bars in enumerate(bar_data) : 
            pt = []
            ## bottom left
            xs = int((bars['x0'])*map_sz/width)
            ys = int((bars['y0'])*map_sz/height)
            xs_norm = bars['x0']/width
            ys_norm = bars['y0']/height
            if xs == map_sz :
                xs-=1
            if ys == map_sz :
                ys-=1
            if xs < map_sz and ys < map_sz :
                canvas[ys,xs] += 1#len(bar_data)
                _ = (xs, ys)
                pt.append([xs, ys])
                pt_norm.append(list((xs_norm, ys_norm)))
                if _ in pnts : 
                    pnts[_].append(ix)
                else : 
                    pnts[_]=[ix]
            else: 
                print('ERROR', xs, ys, map_sz, f)
                print(lod)
            ## top right
            xs2 = int(((bars['x0'])+bars['width'])*map_sz/width)          
            ys2 = int(((bars['y0'])+bars['height'])*map_sz/height)  
            xs2_norm = (bars['x0']+bars['width'])/width
            ys2_norm = (bars['y0']+bars['height'])/height
            if xs2 == map_sz :
                xs2-=1
            if ys2 == map_sz :
                ys2-=1
            if xs2 < map_sz and ys2 < map_sz and xs2 >= 0 and ys2>= 0  and xs < map_sz and ys < map_sz and xs >= 0 and ys>= 0 :
                canvas[ys2,xs2] += 1#len(bar_data)
                _ = (xs2, ys2)
                pt.append([xs2, ys2])
                pt_norm.append(list((xs2_norm, ys2_norm)))
                if _ in pnts : 
                    pnts[_].append(ix)
                else : 
                    pnts[_]=[ix]
                
                canvas[ys,xs2] += 1#len(bar_data)
                _ = (xs2, ys)
                pt.append([xs2, ys])
                pt_norm.append(list((xs2_norm, ys_norm)))
                if _ in pnts : 
                    pnts[_].append(ix)
                else : 
                    pnts[_]=[ix]
                
                canvas[ys2,xs] += 1#len(bar_data)
                _ = (xs, ys2)
                pt.append([xs, ys2])
                pt_norm.append(list((xs_norm, ys2_norm)))
                if _ in pnts : 
                    pnts[_].append(ix)
                else : 
                    pnts[_]=[ix]
            else: 
                print('ERROR', xs, ys, xs2, ys2, map_sz, f)
                print(lod)
            ptss.append(pt)
        for bars in bar_data : 
            interp_on = []
            if bars['width'] == 0 : 
                bars['width'] = 1
            if bars['height'] == 0 : 
                bars['height'] = 1
            ap = bars['x0']*map_sz/width
            bq = bars['y0']*map_sz/height
            cr = (bars['x0']+bars['width'])*map_sz/width          
            ds = (bars['y0']+bars['height'])*map_sz/height 
            # Top hr line 
            interp_on.append([(ap, ds), (cr, ds)])
            # left vert line 
            interp_on.append([(ap, bq), (ap, ds)])
            # right vert line 
            interp_on.append([(cr, bq), (cr, ds)])
            # botom hr line 
            interp_on.append([(ap, bq), (cr, bq)])
            for ix, pts_ in enumerate(interp_on) :
                (i1, i2), (i3, i4) = pts_
                canvas = self.interpolt(canvas, i1, i2, i3, i4, 10)
            # canvas = self.gauss(canvas, ap, ds, map_sz)
            # canvas = self.gauss(canvas, cr, ds, map_sz)
            # canvas = self.gauss(canvas, ap, bq, map_sz)
            # canvas = self.gauss(canvas, cr, bq, map_sz)
        canvas = gaussian_filter(canvas, 1)
        return canvas, self.get_edge_list(ptss, pnts), ptss, pt_norm

    def process_fs(self, js_obj, canvas, width, height, f):
            ln_data = js_obj['data']['curves']
            print('ln_data')
            print(ln_data)
            map_sz = self.map_sz
            ptss = []
            pt_norm = []
            pnts = {}
            for ix, ln in enumerate(js_obj['data']['curves']) :
                xs = ln['x']
                ys = ln['y']
                if type(xs) == float :
                    xs =[[xs]]
                    ys =[[ys]]
                print(xs, type(xs))
                xs_norm = [(xs[i][0])/width for i in range(len(xs))]
                xs = [xs[i][0]*map_sz/width for i in range(len(xs))]
                ys_norm = [(ys[i][0])/height for i in range(len(ys))]
                ys = [ys[i][0]*map_sz/height for i in range(len(ys))]
                ln_pts = [[xs[i], ys[i]] for i in range(len(xs))]
                ln_pts_norm = [[xs_norm[i], ys_norm[i]] for i in range(len(xs))]
                ptss.append(ln_pts)
                for pt_ix, pt_ in enumerate(ln_pts) :
                    x_, y_ = pt_
                    xs_nor, ys_nor = ln_pts_norm[pt_ix]
                    pt_norm.append(list((xs_nor, ys_nor)))
                    if (x_, y_ ) in pnts : 
                        pnts[(x_, y_ )].append(ix)
                    else :
                        pnts.update({(x_, y_ ):[ix]})
                    if x_ < map_sz and y_ <map_sz :
                        canvas[int(y_),int(x_)] += 1

            for ln in js_obj['data']['curves'] :
                xs = ln['x']
                xs = [xs[i][0]*map_sz/width for i in range(len(xs))]
                ys = ln['y']
                ys = [ys[i][0]*map_sz/height for i in range(len(ys))]
                ln_pts = [[xs[i], ys[i]] for i in range(len(xs))]
                for ix, pt_ in enumerate(ln_pts) :
                    x_, y_ = pt_
                    if x_ < map_sz and y_ <map_sz  and ix >0:
                        x_1, y_1 = ln_pts[ix-1]
                        self.interpolt(canvas, x_, y_, x_1, y_1)
                    
            canvas = gaussian_filter(canvas, 1)
            return canvas, self.get_edge_list(ptss, pnts), ptss, pt_norm


    def save(self, f, canvas):
        np.save(osp.join(self.sd,f[:-5]), canvas)
        print('Save success')
        print("*"*80)
        
    def generate_masks(self, chart_type, js_obj, canvas, width, height, f) :
        # print(chart_type, width, height, f, self.mode)
    
        if chart_type == 'line' or chart_type == 'line2' :
            canvas, el, pt, pt_norm= self.process_lines(js_obj, canvas, width, height, f)
        elif chart_type == 'vertical_box' or chart_type == 'vbox' or chart_type == 'hbox':
            canvas, el, pt, pt_norm= self.process_boxplots(js_obj, canvas, width, height, f)
        elif chart_type == 'scatter2':
            canvas, el, pt, pt_norm= self.process_scatter_synth(js_obj, canvas, width, height, f)
        elif chart_type == 'scatter':
            canvas, el, pt, pt_norm= self.process_scatter_pmc(js_obj, canvas, width, height, f)
        elif chart_type == 'horizontal_bar' or chart_type =='hGroup'  or chart_type =='hStack':
            canvas, el, pt, pt_norm= self.process_h_bar(js_obj, canvas, width, height, f)
        elif chart_type == 'vertical_bar'  or chart_type =='vStack'  or chart_type =='vGroup':
            canvas, el, pt, pt_norm= self.process_h_bar( js_obj, canvas, width, height, f)
        elif chart_type == 'figureSeer':
            canvas, el, pt, pt_norm= self.process_fs(js_obj, canvas, width, height, f)
        else :
            print('CHART TYPE NOT FOUND')
            return js_obj, f
        if canvas.sum() == 0 :
            raise ValueError('HM sum 0 Found')    
        
        if self.mode =='save' :
            self.save(f, canvas)        
        elif self.mode =='run' :
            return canvas, el, pt, pt_norm


if __name__ == "__main__":
    pass