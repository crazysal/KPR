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

from scipy.spatial import distance_matrix



def countIslands(mat, th=0):
    # print('In countIslands', mat.shape)
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

def get_collated_components_big(cmps, collate_count):
    cmp_pts_collate = []
    for c in cmps :
        pts_collate = []
        # print('\n', len(c), '---- \t', c)
        if len(c) < collate_count :
            _x0 = []
            _y0 = []
            for _p in c :
                _x0.append(_p[0])
                _y0.append(_p[1])
            # print('colted', _x0, _y0)
            # print('lt 5', int(sum(_x0)/len(_x0)), int(sum(_y0)/len(_y0)), len(pts_collate) )
            pts_collate.append((int(sum(_x0)/len(_x0)), int(sum(_y0)/len(_y0))))
        else : 
            _x0 = []
            _y0 = []
            i = 0
            for _pix in range(0,len(c), 6) :
                while i < collate_count and _pix+i < len(c):
                    _x0.append(c[_pix+i][0])
                    _y0.append(c[_pix+i][1])
                    i+=1
                # print('colted', _x0, _y0)
                # print('mt 5', _pix, int(sum(_x0)/len(_x0)), int(sum(_y0)/len(_y0)), len(pts_collate) )
                pts_collate.append((int(sum(_x0)/len(_x0)), int(sum(_y0)/len(_y0))))
                i =0
                _x0 = []
                _y0 = []
        cmp_pts_collate.append(pts_collate)
    return cmp_pts_collate

def get_collated_points_small(cmps, collate_count):
    pts_collate = []
    for c in cmps :
        if len(c) < collate_count :
            _x0 = []
            _y0 = []
            for _p in c :
                _x0.append(_p[0])
                _y0.append(_p[1])
            pts_collate.append((int(sum(_x0)/len(_x0)), int(sum(_y0)/len(_y0))))
        else : 
            _x0 = []
            _y0 = []
            i = 0
            for _pix in range(0,len(c), collate_count) :
                while i < collate_count and _pix+i < len(c):
                    _x0.append(c[_pix+i][0])
                    _y0.append(c[_pix+i][1])
                    i+=1
                pts_collate.append((int(sum(_x0)/len(_x0)), int(sum(_y0)/len(_y0))))
                i =0
                _x0 = []
                _y0 = []
    return pts_collate

def get_major_minor(cmp_pts_collate, num_to_clst):
    lens_ = [len(_) for _ in cmp_pts_collate]
    print('length of components after collate', lens_)
    cluster_index = np.argpartition(lens_, -num_to_clst)[-num_to_clst:]
    major_clusters = [cmp_pts_collate[i_] for i_ in cluster_index]
    print('maj clusters =', len(major_clusters))
    minor_clusters = [cmp_pts_collate[i_] for i_ in range(len(cmp_pts_collate)) if i_ not in cluster_index]
    print('mino clusters =', len(minor_clusters))
    return major_clusters, minor_clusters

def get_closest_lines(pts_collate):
    pts_collate = np.array(pts_collate)
    dist = distance_matrix(pts_collate, pts_collate)
    dist[dist==0] = float('inf')
    closest = np.argmin(dist, axis=0)
    lines_pair = []
    for p_colt, p_clos in zip(pts_collate, closest) :
        lines_pair.append([tuple(pts_collate[p_clos]), tuple(p_colt)])
    lines = []
    for lp_ix in range(len(lines_pair)) :
        for lp_jx in range(len(lines_pair)) :
            if lp_ix !=lp_jx :
                p1_i, p2_i = lines_pair[lp_ix]
                # print('p1_i, p2_i', p1_i, p2_i)
                p1_j, p2_j = lines_pair[lp_jx]
                # print('p1_j, p2_j', p1_j, p2_j)
                l_i=False 
                l_j=False
                for c in lines :
                    if p1_i in c or p2_i in c:
                        c.add(p1_i)
                        c.add(p2_i)
                        l_i = True
                    if p1_j in c or p2_j in c:
                        c.add(p1_j)
                        c.add(p2_j)
                        l_j = True
                if not l_i :
                    print('Not l_i')
                    for l in lines:
                        print(l)
                    lines.append(set())
                    lines[-1].add(p1_i)
                    lines[-1].add(p2_i)
                for c in lines :
                    if p1_i in c or p2_i in c:
                        c.add(p1_i)
                        c.add(p2_i)
                        l_i = True
                    if p1_j in c or p2_j in c:
                        c.add(p1_j)
                        c.add(p2_j)
                        l_j = True
                if not l_j :
                    print('Not l_j')
                    for l in lines:
                        print(l)
                    lines.append(set())
                    lines[-1].add(p1_j)
                    lines[-1].add(p2_j)
    line_list = []
    for l in lines : 
        # plt.plot([y_ for _, y_ in list(l)], [x_ for x_, _ in list(l)])
        line_list.append(list(l))
    return line_list   

def get_assigned(major_clusters, minor_clusters):
    minor_c_dist = []
    minor_alloted = []
    for cmp_pts_ix, cmp_pt in enumerate(minor_clusters) :
        major_dist= []
        for maj in major_clusters :
            min_dist = float('inf')
            for query in cmp_pt :
                for match in maj :
                    # print(query, match)
                    d = distance.euclidean(query, match)
                    # print(d)
                    min_dist = min(min_dist, d)
            # print(min_dist, '\n')
            major_dist.append(min_dist)
        minor_c_dist.append(major_dist)
    for mins in minor_c_dist :
        minor_alloted.append(mins.index(min(mins)))
    for _cix, c in enumerate(minor_clusters) :
        label = minor_alloted[_cix]
        for c_ in c : 
            major_clusters[label].append(c_)
    print('maj', len(major_clusters))
    return major_clusters

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
            
def get_canvas(ln_data, height, width, map_sz) :
    canvas = torch.zeros((128,128))
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
    return canvas
def get_xy():
    json_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/json_type/line'
    hm_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/hms/9'
    img_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/images/'
    sv_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/json/9/6a_t2:20'
    for h in os.listdir(hm_dir) :
        print('_'*80)
        print('start', h)
        hmx = 1
        fig = plt.figure(figsize=(30, 14))
        fig.add_subplot(4,5,hmx)
        hmx+=1
        im = Image.open(osp.join(img_dir, h[:-4]+'.jpg'))
        width, height = im.size
        plt.title("Input Chart")
        plt.imshow(im)
        js_obj = json.load(open(osp.join(json_dir, h[:-4]+'.json'), 'r'))
        ln_data = js_obj['task6']['output']['visual elements']['lines']
        canvas = get_canvas(ln_data, height, width, 128)
        fig.add_subplot(4,5,hmx)
        hmx+=1
        plt.title("GT Mask")
        plt.imshow(canvas)
        hm = np.load(osp.join(hm_dir, h))
        mse3 = hm[2][0][1]
        mse3 = F.sigmoid(torch.from_numpy(mse3)).numpy()
        fig.add_subplot(4,5,hmx)
        hmx+=1
        plt.title("Pred Heatmap MSE")
        plt.imshow(mse3)
        x_, y_ = np.unravel_index(np.argsort(mse3.ravel()), mse3.shape)
        lgnds = js_obj['task6']['input']['task5_output']['legend_pairs']
        num_to_clst = len(lgnds) ## get from legend gt or else 1
        if num_to_clst == 0 : 
            num_to_clst = 1
        print('legend found',len(lgnds),  num_to_clst)
        threshold1 = 250
        threshold2 = 20
        collate_count = threshold2       
        
        x_1, y_1 = x_[-threshold1:], y_[-threshold1:]
        canvas = np.zeros(mse3.shape)
        for x, y in zip(x_1, y_1) :
            canvas[x][y] = 1 
        fig.add_subplot(4,5,hmx)
        hmx+=1
        plt.title("Selected points Threshold by max probablity")
        plt.imshow(canvas)
        cmps = countIslands(canvas)
        ## Direct component
        cmp_pts_collate = get_collated_components_big(cmps, collate_count)   
        # ## Fine grained
        # pts_collate = get_collated_points_small(cmps, collate_count)
        # cmp_pts_collate = get_closest_lines(pts_collate)
       
        if len(cmp_pts_collate) < num_to_clst :
            num_to_clst = len(cmp_pts_collate)
        ### Get Major/Minor
        major_clusters, minor_clusters = get_major_minor(cmp_pts_collate, num_to_clst)

        print(major_clusters, minor_clusters)
        ### Assign Minor 
        major_clusters =  get_assigned(major_clusters, minor_clusters)
        print(major_clusters)
        plt.show()

        exit()
        ### Convert to json format image coord
        op_obj = {"task6": {"output": { "data series": [] ,"visual elements": {"lines": []}}}}
        
        for k in range(len(major_clusters)):
            coord_arr = []
            for j in range(len(major_clusters[k])):
                coord_arr.append({"x" : major_clusters[k][j][0]*width/128, "y" : major_clusters[k][j][1]*height/128})
            op_obj["task6"]["output"]["visual elements"]["lines"].append(coord_arr)
        with open(osp.join(sv_dir, h[:-4]+'.json'), 'w') as f_:
            json.dump(op_obj, f_)
        print('Saved', op_obj["task6"]["output"]["visual elements"]["lines"])


if __name__ == '__main__' :
    get_xy()

'''

a = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/json/9/6a'
b = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/json_type/line/'
python metric6a.py /home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/heat/json/9/6a /home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_4/json_type/line/
for m in major_clusters : 
    plt.plot([y_ for _, y_ in m], [x_ for x_, _ in m])

///
can2 = np.zeros(mse3.shape)
for x, y in pts_collate :
    can2[x][y] = 1 
    

print(canvas.sum()); plt.imshow(canvas); plt.show()

print(can2.sum()); plt.imshow(can2); plt.show()

import numpy as np




img_clst =Image.new("RGB", (128,128))
im_ = ImageDraw.Draw(img_clst)
for lp in lines_pair :
    im_.line(lp, fill='red', width=1)
'''