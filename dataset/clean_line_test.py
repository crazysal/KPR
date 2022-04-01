import os 
import json 
import PIL 
from PIL import Image, ImageDraw, ImageStat
import pickle
json_dir = '/home/sahmed9/reps/KeyPointRelations/data/line_json_test'
im_dir= '/home/sahmed9/reps/KeyPointRelations/data/images/test_images_pmc_line'
sv_dir = '/home/sahmed9/reps/KeyPointRelations/data/images/test_pmc_line_clean/'

json_dir = '/home/sahmed9/reps/KeyPointRelations/data/JSONs/line'
im_dir= '/home/sahmed9/reps/KeyPointRelations/data/images/line/'
sv_dir = '/home/sahmed9/reps/KeyPointRelations/data/images/line_clean/'
fl = '/home/sahmed9/reps/KeyPointRelations/data/split4_train_task6_line.pkl'

# json_dir = '/home/sahmed9/reps/KeyPointRelations/data/JSONs/line2'
# im_dir= '/home/sahmed9/reps/KeyPointRelations/data/images/line2/'
# sv_dir = '/home/sahmed9/reps/KeyPointRelations/data/images/line2_clean/'
# fl = '/home/sahmed9/reps/KeyPointRelations/data/split4_train_task6_line2.pkl'

f = open(fl, 'rb')
f = pickle.load(f)
print(f['line'][0])
js = f['line']
# js = os.listdir(json_dir)
# for i in range(len(js)):
    # im_nm = f[i][:-4]+'jpg'
for i in range(len(f['line'])):
    im_nm = f['line'][i][:-4]+'jpg'
    # im_nm = f['line2'][i][:-4]+'png'
    print('*'*80)
    print(im_nm)
    im = Image.open(os.path.join(im_dir, im_nm))
    js_obj = json.load(open(os.path.join(json_dir, js[i]), 'r')) 
    print(js_obj)
    imd = ImageDraw.Draw(im)
    if js_obj['task6']['input']['task4_output'] is not None :
        ploth, plotw , x0, y0 = js_obj['task6']['input']['task4_output']['_plot_bb'].values()
        ctp = js_obj['task6']['input']['task1_output']['chart_type'] 
        tb =  js_obj['task6']['input']['task2_output']['text_blocks']
        lp =  js_obj['task6']['input']['task5_output']['legend_pairs']
        ln_data = js_obj['task6']['output']['visual elements']['lines']
        x_axis = js_obj['task6']['input']['task4_output']['axes']['x-axis']
        y_axis = js_obj['task6']['input']['task4_output']['axes']['y-axis']
        for pt in x_axis :
            x_, y_ = pt['tick_pt']['x'], pt['tick_pt']['y']
            shape = [(x_, y_), (x_+2, y_+5)]
            print('axis tb', [(x_, y_), (x_+2, y_+5)])
            cl = ImageStat.Stat(im).median
            imd.rectangle(shape, fill =tuple(cl),outline='green')
        for pt in y_axis :
            x_, y_ = pt['tick_pt']['x'], pt['tick_pt']['y']
            shape = [(x_, y_), (x_+5, y_+2)]
            print('axis tb', [(x_, y_), (x_+2, y_+5)])
            cl = ImageStat.Stat(im).median
            imd.rectangle(shape, fill =tuple(cl),outline='green')
        ## remove text box
        for bx in tb :
            poly = bx['polygon'] if 'polygon' in bx else bx['bb']                
            # print(poly)
            x_min = min(int(poly['x0']), int(poly['x1']), int(poly['x2']), int(poly['x3']))
            x_max = max(int(poly['x0']), int(poly['x1']), int(poly['x2']), int(poly['x3']))
            y_min = min(int(poly['y0']), int(poly['y1']), int(poly['y2']), int(poly['y3']))
            y_max = max(int(poly['y0']), int(poly['y1']), int(poly['y2']), int(poly['y3']))
            # print(x_min,x_max, y_min,y_max)
            # img_[y_min:y_max, x_min :x_max, :] = 255
            shape = [(x_min, y_min), (x_max, y_max)]
            print('removed tb', [(x_min, y_min), (x_max, y_max)])
            cl = ImageStat.Stat(im).median
            imd.rectangle(shape, fill =tuple(cl),outline='blue')
        ## remove legend
        for bx in lp : 
            x_min = bx['bb']['x0']
            y_min = bx['bb']['y0']
            x_max = bx['bb']['x0'] + bx['bb']['width']
            y_max = bx['bb']['y0']+ bx['bb']['height']
            # img_[y_min:y_max, x_min :x_max, :] = 255 
            shape = [(x_min, y_min), (x_max, y_max)]
            print('removed leg', [(x_min, y_min), (x_max, y_max)])
            cl = ImageStat.Stat(im).median
            imd.rectangle(shape, fill =tuple(cl),outline='red')
        print('crop', (x0, y0, x0+plotw, y0+ploth))
        im = im.crop((x0, y0, x0+plotw, y0+ploth))
        im.save(os.path.join(sv_dir, im_nm))

# 'PMC4132901___12889_2014_6915_Fig1_HTML.json'
# outline='red'