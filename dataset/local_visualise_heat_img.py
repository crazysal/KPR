from collections import deque
from scipy.spatial import distance
import os.path as osp
import numpy as np 
import json 
import os 
import matplotlib 
from matplotlib import pyplot as plt 
import PIL 
import PIL.Image as Image
import torch 
import torch.nn as nn 
import torch.nn.functional as F  
from scipy.interpolate import interp1d

from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
## Edge list 
'''
h in hms
hm(1, 5, 128, 128)
hm(1, 5, 128, 128)
hm(1, 5, 128, 128)
node(47, 128)
ecl(243, 2)
eemb(243, 256)
'''
def visualise_heat_img():
    map_sz = 128
    ## PMC LINE Test 
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
    # hm_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/hms/9/'
    # op_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/oput/9/'
    hm_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/hms/10b_line_test/'
    op_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/oput/10b/'
    ### DAVE 
    # img_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/test_images_dave/'
    # json_dir = None
    # hm_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/hms/9dave/'
    # op_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/oput/9dave/'
    #### PMC scatter train
    # img_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/scatter/'
    # json_dir = None #'/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/scatter/'
    # hm_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/hms/9_scatter'
    # op_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/oput/9_scatter'
    # ## Train Ground Truth 
        # img_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/line/'
        # hm_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/line_mask/'
        # op_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/gt/'
    if not osp.isdir(op_dir) :
        os.mkdir(op_dir)
    for hm in os.listdir(hm_dir) :
        print('..')
        print('hm', hm)
        print('hm 4', hm[:-4])
        print('hm j', hm[:-5]+'jpg')
        print('hm d', hm[:-4]+'json')
            
        ## image
        fig = plt.figure(figsize=(30, 14))
        fig.add_subplot(4,5,1)
        img = plt.imread(osp.join(img_dir, hm[:-4]+'jpg'))
        plt.title("Input Chart")
        plt.imshow(img)
        if json_dir is not None :
            ## LAbel;
            fig.add_subplot(4,5,2)
            canvas = torch.zeros((128,128))
            print(img.shape)
            ims = img.shape
            height, width = ims[0], ims[1]
            f =  hm[:-4]+'json'
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
            fig.add_subplot(4,5,3)
            canvas[canvas >=1] =1
            canvas[canvas <1] = 0 
            plt.title("GT Heatmap Classification")
            plt.imshow(canvas)
            print('*'*80)
        ### Predictions 
        pred = np.load(osp.join(hm_dir, hm), allow_pickle = True)
        print('got pred', pred.shape)
        hmx = 6
        # for hm_ in pred :
        for i in range(3) :
            hm_ = pred[i]
            hm_ = hm_.squeeze(0)
            print(i, 'pred', hm_.shape)
            fm_mse = hm_[0:2, :, :]
            fm_bce = hm_[3, :, :]
            fm_ce  = torch.softmax(torch.from_numpy(hm_[3:5, :, :]),dim=1)
            hm_a = F.sigmoid(torch.from_numpy(fm_mse[0])).numpy()
            hm_b = F.sigmoid(torch.from_numpy(fm_mse[1])).numpy()
            fig.add_subplot(4,5,hmx)
            hmx+=1
            plt.title("Pred Heatmap MSE probablity c1")
            plt.imshow(hm_a)
            print('->', hm, np.sum(hm_))       
            fig.add_subplot(4,5,hmx)
            hmx+=1
            plt.title("Pred Heatmap MSE probablity c2")
            plt.imshow(hm_b)
            print('->', hm_.shape, np.sum(hm_))       
            ## BCE
            hm_2a = F.sigmoid(torch.from_numpy(fm_bce)).numpy()
            # hm_2b = F.sigmoid(torch.from_numpy(hm_2[1])).numpy()
            fig.add_subplot(4,5,hmx)
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
            fig.add_subplot(4,5,hmx)
            hmx+=1
            plt.title("Pred Heatmap classification c1")
            plt.imshow(fm_ce[0])
            # print('-> hm2', np.sum(hm_3[0]))
            fig.add_subplot(4,5,hmx)
            hmx+=1
            plt.title("Pred Heatmap classification c2")
            plt.imshow(fm_ce[1])
            # print('->', hm_3.shape, np.sum(hm_3[1]))       
            #
        plt.savefig(osp.join(op_dir, hm[:-3]+'jpg'))
            # break
    plt.close('all')

