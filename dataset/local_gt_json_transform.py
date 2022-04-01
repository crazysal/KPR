## Line Hardest
im = Image.open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/line/PMC5506989___materials-10-00380-g005.jpg')
js_obj = json.load(open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/line/PMC5506989___materials-10-00380-g005.json', 'r'))
hm = np.load('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/all_pmc_mask_plotbb_128/line/PMC5506989___materials-10-00380-g005.npy')
## Line Hardest2
im = Image.open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/line/PMC5554038___materials-10-00657-g005.jpg')
js_obj = json.load(open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/line/PMC5554038___materials-10-00657-g005.json', 'r'))
hm = np.load('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/all_pmc_mask_plotbb_128/line/PMC5554038___materials-10-00657-g005.npy')
## Line Easy
im = Image.open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/line/PMC1164420___1471-2458-5-51-1.jpg')
js_obj = json.load(open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/line/PMC1164420___1471-2458-5-51-1.json', 'r'))
hm = np.load('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/all_pmc_mask_plotbb_128/line/PMC1164420___1471-2458-5-51-1.npy')
#Scater easy
im = Image.open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/scatter/PMC5666481___nanomaterials-07-00316-g002.jpg')
js_obj = json.load(open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/scatter/PMC5666481___nanomaterials-07-00316-g002.json', 'r'))
hm = np.load('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/all_pmc_mask_plotbb_128/scatter/PMC5666481___nanomaterials-07-00316-g002.npy')
#Scater hard
im = Image.open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/scatter/PMC5503364___materials-10-00299-g007.jpg')
js_obj = json.load(open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/scatter/PMC5503364___materials-10-00299-g007.json', 'r'))
hm = np.load('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/all_pmc_mask_plotbb_128/scatter/PMC5503364___materials-10-00299-g007.npy')
#Vertical box hard
im = Image.open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/vertical_box/PMC5295288___ijerph-14-00037-g003.jpg')
js_obj = json.load(open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/vertical_box/PMC5295288___ijerph-14-00037-g003.json', 'r'))
hm = np.load('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/all_pmc_mask_plotbb_128/vertical_box/PMC5295288___ijerph-14-00037-g003.npy')

#Horizontal bar hard
im = Image.open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/horizontal_bar/PMC4854880___fpubh-04-00087-g002.jpg')
js_obj = json.load(open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/horizontal_bar/PMC4854880___fpubh-04-00087-g002.json', 'r'))
hm = np.load('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/all_pmc_mask_plotbb_128/horizontal_bar/PMC4854880___fpubh-04-00087-g002.npy')

im = Image.open('/home/sahmed9/Documents/data/charts/FigureSeer/FigureSeerDataset/Annotated images/03320-10.1.1.164.2102-Figure-3.png')
js_obj = json.load(open('//home/sahmed9/Documents/data/charts/FigureSeer/FigureSeerDataset/Annotations/03320-10.1.1.164.2102-Figure-3.json', 'r'))
w, h = im.size
width, height = w, h 
sz = 128
map_sz = sz
a = np.zeros((sz, sz))
canvas = a 

pt = []
pnts = {}

#Figure seer 

ln_data = js_obj['data']['curves']
        map_sz = self.map_sz
pt = []
pnts = {}
for ix, ln in enumerate(js_obj['data']['curves']) :
    xs = ln['x']
    print('xs', xs)
    xs = [xs[i][0]*map_sz/width for i in range(len(xs))]
    print('xs',xs)
    ys = ln['y']
    print('ys',ys)
    ys = [ys[i][0]*map_sz/height for i in range(len(ys))]
    print('ys',ys)
    ln_pts = [[xs[i], ys[i]] for i in range(len(xs))]
    print('ln_pts', ln_pts)
    pt += ln_pts
    print('pt', pt)
    for pt_ in ln_pts :
        x_, y_ = pt_
        if (x_, y_ ) in pnts : 
            pnts[(x_, y_ )].append(ix)
        else :
            pnts.update({(x_, y_ ):[ix]})
        if x_ < map_sz and y_ <map_sz :
            canvas[int(y_),int(x_)] += 1

hm = copy.deepcopy(a)


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
            interpolt(canvas, x_, y_, x_1, y_1)




def gauss(a, px128, py128, sz):
    for x in range(sz) :
        for y in range(sz) :
            vala = round(math.exp(-((x-px128)**2 + (y-py128)**2))/4, 1)
            if vala > 0 :
                a[y, x] += vala    


def interpolt(a, x1,y1, x2,y2, p=10) :
    print('got values', x1,y1, x2,y2)
    r = max(x2, x1) - min(x2, x1)
    if r != 0 : 
        print('x', round(r,3), round(r/p, 3))
        z = interp1d([x1,x2],[y1, y2],fill_value="extrapolate")
        print(z)
        xs = np.arange(min(x2, x1), max(x2, x1), r/p)
        print(xs)
        ys = z(xs)
        print(ys)
        for ix__, x in enumerate(xs):
            a[int(ys[ix__]), int(x)] =1
    else :
        r = max(y2, y1) - min(y2, y1)
        print('y', round(r, 3), round(r/p, 3))
        z = interp1d([y1, y2], [x1,x2], fill_value="extrapolate")
        print(z)
        ys = np.arange(min(y2, y1), max(y2, y1), r/p)
        print(ys)
        xs = z(ys)
        print(xs)
        for iy__, y in enumerate(ys):
            a[int(y), int(xs[iy__])] =1



### Bar Charts
bar_data = js_obj['task6']['output']['visual elements']['bars']
    
for bars in bar_data : 
    interp_on = []
    print('\n\n')
    if bars['width'] == 0 : 
        bars['width'] = 1
    if bars['height'] == 0 : 
        bars['height'] = 1
    ap = bars['x0']*map_sz/width
    bq = bars['y0']*map_sz/height
    cr = (bars['x0']+bars['width'])*map_sz/width          
    ds = (bars['y0']+bars['height'])*map_sz/height 
    # print('****', (ap, bq), (cr, ds), bars['width'], bars['height'])
    # Top hr line 
    interp_on.append([(ap, ds), (cr, ds)])
    # left vert line 
    interp_on.append([(ap, bq), (ap, ds)])
    # right vert line 
    interp_on.append([(cr, bq), (cr, ds)])
    # botom hr line 
    interp_on.append([(ap, bq), (cr, bq)])
    # print("###", interp_on)
    for ix, pts_ in enumerate(interp_on) :
        (i1, i2), (i3, i4) = pts_
        # print('++++++', ix, i1, i2, i3, i4)
        interpolt(a, i1, i2, i3, i4, 10)
    gauss(canvas, ap, ds, map_sz)
    gauss(canvas, cr, ds, map_sz)
    gauss(canvas, ap, bq, map_sz)
    gauss(canvas, cr, bq, map_sz)

for ix, bars in enumerate(bar_data) : 
    ## bottom left
    xs = int((bars['x0'])*map_sz/width)
    ys = int((bars['y0'])*map_sz/height)
    if xs == map_sz :
        xs-=1
    if ys == map_sz :
        ys-=1
    if xs < map_sz and ys < map_sz :
        canvas[ys,xs] += len(bar_data)
        _ = (xs, ys)
        pt.append([xs, ys])
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
    if xs2 == map_sz :
        xs2-=1
    if ys2 == map_sz :
        ys2-=1
    if xs2 < map_sz and ys2 < map_sz :
        canvas[ys2,xs2] += len(bar_data)
        _ = (xs2, ys2)
        pt.append([xs2, ys2])
        if _ in pnts : 
            pnts[_].append(ix)
        else : 
            pnts[_]=[ix]
        
        canvas[ys,xs2] += len(bar_data)
        _ = (xs2, ys)
        pt.append([xs2, ys])
        if _ in pnts : 
            pnts[_].append(ix)
        else : 
            pnts[_]=[ix]
        
        canvas[ys2,xs] += len(bar_data)
        _ = (xs2, ys)
        pt.append([xs2, ys])
        if _ in pnts : 
            pnts[_].append(ix)
        else : 
            pnts[_]=[ix]
    else: 
        print('ERROR', xs, ys, map_sz, f)
        print(lod)

## Box Plots 
#'first_quartile'
#'max'
#'median'
#'min'
#'third_quartile'


box_data = js_obj['task6']['output']['visual elements']['boxplots']




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
        # print(x1, y1, x2, y2)
        interpolt(a, x1, y1, x2, y2, 10)
            
    gauss(canvas, mx['x'], mx['y'], map_sz)
    gauss(canvas, fq['x'], fq['y'], map_sz)
    gauss(canvas, med['x'], med['y'], map_sz)
    gauss(canvas, tq['x'], tq['y'], map_sz)
    gauss(canvas, mn['x'], mn['y'], map_sz)
    


for ix, box in enumerate(box_data) : 
    for pt_ in  box: 
        xs = int((box[pt_]['x'])*map_sz/width)
        ys = int((box[pt_]['y'])*map_sz/height)
        print(pt,box[pt_]['x'], box[pt_]['y'] ,'->', xs, ys )
        if xs == map_sz :
            xs-=1
        if ys == map_sz :
            ys-=1
        if xs < map_sz and ys < map_sz :
            canvas[ys,xs] += len(box_data)
            _ = (xs, ys)
            pt.append([xs, ys])
            if _ in pnts : 
                pnts[_].append(ix)
            else : 
                pnts[_]=[ix]

## Scatter

pt_data = js_obj['task6']['output']['visual elements']['scatter points']
print(len(pt_data))
for pts in pt_data : 
    for pt_ in pts: 
        px128 = int((pt_['x'])*map_sz/width)-1
        py128 = int((pt_['y'])*map_sz/height)-1
        gauss(a, px128, py128, sz)

print(len(pt_data))
for ix, pts in enumerate(pt_data) : 
    for pt_ in pts: 
        px128 = int((pt_['x'])*map_sz/width)-1
        py128 = int((pt_['y'])*map_sz/height)-1
        if px128 < map_sz and py128 < map_sz :
            canvas[py128,px128] += len(pt_data)
            _ = (px128, py128)
            pt.append([px128, py128])
            if _ in pnts : 
                pnts[_].append(ix)
            else : 
                pnts[_]=[ix]
        else: 
            print('ERROR', px128, py128, map_sz, f)   
            print(lod)




### Lines 
ln_data = js_obj['task6']['output']['visual elements']['lines']
 
for ln_ix, ln in enumerate(ln_data) : 
    for ix, pt_ in enumerate(ln) :
        px128 =int(pt_['x']*sz/w)
        py128 =int(pt_['y'] *sz/h)    
        a[py128, px128] += 1 #len(ln_data)
        _ = (px128, py128)
        pt.append([px128, py128])
        if _ in pnts : 
            pnts[_].append(ln_ix)
        else : 
            pnts[_]=[ln_ix]
        
        # print(int(px128), int(py128), pt['x'], pt['y'], a[int(py128), int(px128)] )

for ln in ln_data : 
    for ix, pt_ in enumerate(ln) :
        px128 = pt_['x']*sz/w; py128 =pt_['y'] *sz/h
        if ix >0 :
            ppx128 = ln[ix-1]['x']*sz/w; ppy128 =ln[ix-1]['y'] *sz/h
            interpolt(a, ppx128, ppy128, px128, py128)

for ln in ln_data : 
    for ix, pt_ in enumerate(ln) :
        px128 = pt_['x']*sz/w; py128 =pt_['y'] *sz/h
        gauss(a, px128, py128,sz)
        
        

#################
# Triangle
#######

points = np.array(pt)
tri = Delaunay(points)
tring = points[tri.simplices]
edge_list = []
edge_count = {}
for t in tring : 
    print('\n')
    p1, p2, p3 = t
    print(p1, p2, p3)
    s1, s2, s3 = set(pnts[tuple(p1)]), set(pnts[tuple(p2)]), set(pnts[tuple(p3)])
    print(s1, s2, s3)
    e1, e2, e3 = s1.intersection(s2), s1.intersection(s3), s2.intersection(s3)
    print(e1, e2, e3)
    if len(e1) == 0 :
        edge_list.append([list(p1), 0, list(p2)])
        if -99 in edge_count :
            edge_count[-99] +=1
        else : 
            edge_count[-99] =1
    else:
        edge_list.append([list(p1), 1, list(p2)])
        _ = e1.pop()
        if _ in edge_count :
            edge_count[_] +=1
        else : 
            edge_count[_] =1
    if len(e2) == 0 :
        edge_list.append([list(p1), 0, list(p3)])
        if -99 in edge_count :
            edge_count[-99] +=1
        else : 
            edge_count[-99] =1
    else:
        edge_list.append([list(p1), 1, list(p3)])
        _ = e2.pop()
        if _ in edge_count :
            edge_count[_] +=1
        else : 
            edge_count[_] =1
    if len(e3) == 0 :
        edge_list.append([list(p2), 0, list(p3)])
        if -99 in edge_count :
            edge_count[-99] +=1
        else : 
            edge_count[-99] =1
    else:
        edge_list.append([list(p2), 1, list(p3)])
        _ = e3.pop()
        if _ in edge_count :
            edge_count[_] +=1
        else : 
            edge_count[_] =1




i = 0 
node_dict = {}
for e in edge_list : 
    p1, e_, p2 = e 
    if tuple(p1) not in node_dict : 
        node_dict.update({tuple(p1):i})
        i+=1
    if tuple(p2) not in node_dict : 
        node_dict.update({tuple(p2):i})
        i+=1

## Create clusters from edges 
clusters = []
for e in edge_list : 
    p1, e_, p2 = e 
    p1ix = node_dict[tuple(p1)]
    p2ix = node_dict[tuple(p2)]
    if e_==0 : #  different line
        flag1 = False
        flag2 = False
        for c in clusters : # already encountered line
            if p1ix in c :
                flag1 = True
            if p2ix in c :
                flag2 = True
        if not flag1 :
            clusters.append({p1ix})
        if not flag2 :
            clusters.append({p2ix})


for e in edge_list : 
    p1, e_, p2 = e 
    p1ix = node_dict[tuple(p1)]
    p2ix = node_dict[tuple(p2)]
    if e_==1 : #  same line
        p1_in_c = []
        p2_in_c = []
        sc = set()
        for c_ix, c in enumerate(clusters) : # already encountered line
            if p1ix in c :
                p1_in_c.append(c_ix)
            if p2ix in c :
                p2_in_c.append(c_ix)
        to_merge = set(p1_in_c + p2_in_c)
        for cix in to_merge :
            sc = sc.union(clusters[cix])
        sc.add(p1ix)
        sc.add(p2ix)
        other = set(np.arange(len(clusters))).symmetric_difference(to_merge)
        new_clusters = [clusters[ix__] for ix__ in other]
        new_clusters.append(sc)
        clusters = copy.deepcopy(new_clusters)


clrs = ['AliceBlue', 'DarkOliveGreen', 'Indigo', 'MediumPurple', 'Purple', 'AntiqueWhite', 'DarkOrange', 'Ivory', 'MediumSeaGreen', 'Aqua', 'DarkOrchid', 'Khaki', 'MediumSlateBlue', 'RosyBrown', 'AquaMarine', 'Lavender', 'MediumSpringGreen', 'RoyalBlue', 'Azure', 'DarkSalmon', 'LavenderBlush', 'MediumTurquoise', 'SaddleBrown', 'Beige', 'DarkSeaGreen', 'LawnGreen', 'MediumVioletRed', 'Salmon', 'Bisque', 'DarkSlateBlue', 'LemonChiffon', 'MidnightBlue', 'SandyBrown', 'DarkSlateGray', 'LightBlue', 'MintCream', 'SeaGreen', 'BlanchedAlmond', 'DarkTurquoise', 'LightCoral', 'MistyRose', 'SeaShellBlue', 'DarkViolet', 'LightCyan', 'Moccasin', 'Sienna', 'BlueViolet', 'DeepPink', 'LightGoldenrodYellow', 'NavajoWhite', 'Silver', 'Brown', 'DeepSkyBlue', 'LightGray', 'Navy', 'SkyBlue', 'BurlyWood', 'DimGray', 'LightGreen', 'OldLace', 'SlateBlue', 'CadetBlue', 'DodgerBlue', 'LightPink', 'Olive', 'SlateGray', 'Chartreuse', 'FireBrick', 'LightSalmon', 'OliveDrab', 'Snow', 'Chocolate', 'FloralWhite', 'LightSeaGreen', 'Orange', 'SpringGreen', 'Coral', 'ForestGreen', 'LightSkyBlue', 'OrangeRed', 'SteelBlue', 'CornFlowerBlue', 'Fuchsia', 'LightSlateGray', 'Orchid', 'Tan', 'Cornsilk', 'Gainsboro', 'LightSteelBlue', 'PaleGoldenRod', 'Teal', 'Crimson', 'GhostWhite', 'LightYellow', 'PaleGreen', 'Thistle', 'Cyan', 'Gold', 'Lime', 'PaleTurquoise', 'Tomato', 'DarkBlue', 'GoldenRod', 'LimeGreen', 'PaleVioletRed', 'Turquoise', 'DarkCyan', 'Gray', 'Linen', 'PapayaWhip', 'Violet', 'DarkGoldenRod', 'Green', 'Magenta', 'PeachPuff', 'Wheat','DarkGray', 'GreenYellow', 'Maroon', 'Peru', 'White', 'DarkGreen', 'HoneyDew', 'MediumAquaMarine', 'Pink', 'WhiteSmoke', 'DarkKhaki', 'HotPink', 'MediumBlue', 'Plum', 'Yellow', 'DarkMagenta', 'IndianRed', 'MediumOrchid', 'PowderBlue', 'YellowGreen']

img_ =Image.new("RGB", (128,128))
im_ = ImageDraw.Draw(img_)
for e in edge_list : 
    p1, e_, p2 = e 
    if e_ == 0 : 
        f="red"
    else:
        f="green"
    im_.line([tuple(p1), tuple(p2)], fill=f, width=0)

img_r =Image.new("RGB", (128,128))
im_ = ImageDraw.Draw(img_r)
for e in edge_list : 
    p1, e_, p2 = e 
    if e_ == 0 : 
        f="red"
        im_.line([tuple(p1), tuple(p2)], fill=f, width=0)
    else:
        f="green"

img_b =Image.new("RGB", (128,128))
im_ = ImageDraw.Draw(img_b)
for e in edge_list : 
    p1, e_, p2 = e 
    if e_ == 0 : 
        f="red"
    else:
        f="green"
        im_.line([tuple(p1), tuple(p2)], fill=f, width=0)

rev_node_dict = dict((v,k) for k,v in node_dict.items())
img_clst =Image.new("RGB", (128,128))
im_ = ImageDraw.Draw(img_clst)
for ci_, c in enumerate(clusters) : 
    pt = [rev_node_dict[_] for _ in c]
    # print(len(pt), pt)
    if len(pt) == 1 :
        # im_.point(pt, fill=clrs[0])
        im_.rectangle([pt[0], (pt[0][0]+1, pt[0][1]+1)], fill=clrs[0], width=0)
    else :
        im_.line(pt, fill=clrs[np.random.randint(0, len(clrs))], width=0)

c_rest = sorted(clusters, key=lambda k:len(k), reverse = True)
img_clst_err =Image.new("RGB", (128,128))
im_ = ImageDraw.Draw(img_clst_err)
for ci_, c in enumerate(c_rest[len(ln_data):]) : 
    pt = [rev_node_dict[_] for _ in c]
    # print(len(pt), pt)
    if len(pt) == 1 :
        # im_.point(pt, fill=clrs[0])
        im_.rectangle([pt[0], (pt[0][0]+1, pt[0][1]+1)], fill='Red', width=0.2)
    else :
        im_.line(pt, fill=clrs[ci_+1], width=0)



f, axarr = plt.subplots(2,4)
axarr[0,0].imshow(np.array(im))
axarr[0,1].imshow(hm)
axarr[0,2].imshow(a)
axarr[0,3].imshow(img_)
axarr[1,0].imshow(img_r)
axarr[1,1].imshow(img_b)
axarr[1,2].imshow(img_clst)
axarr[1,3].imshow(img_clst_err)
# axarr[1,1].triplot(points[:,0], points[:,1], tri.simplices)
# axarr[1,1].plot(points[:,0], points[:,1], 'o')

plt.show()


##plot Clusters 


# Setting false 
cluster_to_cluster = [[True for i in range(0, len(clusters))] for i in range(0, len(clusters))]
for e in edge_list : 
    p1, e_, p2 = e 
    if not e_: ## cluster can not be merged
        p1ix = node_dict[tuple(p1)]
        p2ix = node_dict[tuple(p2)]
        c1ix = []
        c2ix = []
        print('\n', p1ix, p2ix, c1ix, c2ix)
        for cix, c in enumerate(clusters) :
            if p1ix in c : 
                print('$$', cix)
                c1ix.append(cix)
            if p2ix in c : 
                print('##', cix)
                c2ix.append(cix)
        print(p1ix, p2ix, c1ix, c2ix)
        cluster_to_cluster[c1ix[0]][c2ix[0]] = False
        cluster_to_cluster[c1ix[0]][c1ix[0]] = False
        cluster_to_cluster[c2ix[0]][c1ix[0]] = False
        cluster_to_cluster[c2ix[0]][c2ix[0]] = False

## Actual Merging
agg_clust = []
for c1ix, c1 in enumerate(cluster_to_cluster) :
    super_cluster = set()
    print('\n', c1ix, len(c1), len(clusters[c1ix]))
    super_cluster = super_cluster.union(clusters[c1ix])
    print(len(super_cluster))
    for c2ix, c2 in enumerate(c1) :
        print('#', c2, len(super_cluster), len(clusters[c2ix]))
        if c2 : 
            super_cluster = super_cluster.union(clusters[c2ix])
        print(':', len(super_cluster))
    flag = False
    for ac in agg_clust :
        print(ac)
        if ac == super_cluster :
            flag = True
    if not flag :
        print('---', len(super_cluster))
        agg_clust.append(super_cluster)
    break

sc = set()         
## Sanity check         
for i in range(len(clusters)) :
    for j in range(len(clusters)) :
        sc = sc.union(clusters[i]).union(clusters[j])
        if clusters[i] == clusters[j] :
            print(i, j, len(clusters[i]), len(clusters[j]), len(clusters[i].union(clusters[j])))
            # if i !=j :





same, diff = [], []
for e in edge_list : 
    p1, e_, p2 = e 
    if e_ == 0 : 
        diff.append((node_dict[tuple(p1)], node_dict[tuple(p2)]))
    else:
        same.append((node_dict[tuple(p1)], node_dict[tuple(p2)]))


sorted(node_dict.items() , key= lambda kv:kv[1] )

