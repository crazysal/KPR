import PIL 
import PIL.Image as Image 
import json 
from matplotlib import pyplot as plt 
import plotly
im = Image.open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/line/PMC5554038___materials-10-00657-g005.jpg')
js_obj = json.load(open('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/line/PMC5554038___materials-10-00657-g005.json', 'r'))

# 
class Plot_LINES():
    def __init__(self, **kwargs):
            self.mode = kwargs['mode']
            self.choice = kwargs['ch']
            self.seed = 42
            np.random.seed(self.seed)
            js_dir = '/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations_JSON/line/'
            self.js_objs = self.get_js(js_dir)
            
    @staticmethod
def get_plt_fmt(num, type_='line') :
    mrkrs = ['.',',','o','v','^','<','>','1','2','3','4','8','s','p','P','*','h','H','+','x','X','D','d','|','_']
    if type_ == 'line' :
        nn = [None]*2*len(mrkrs)
        mrkrs+=nn
    # ln_sty = ['-','--','-.',':', None]
    ln_sty = ['-','--','-.',':']
    clrs = ['b','g','r','c','m','y','k',None]
    fmt = []
    for i in range(num):
        m_ix = np.random.randint(0, len(mrkrs))
        s_ix = np.random.randint(0, len(ln_sty))
        c_ix = np.random.randint(0, len(clrs))
        _m = mrkrs[m_ix] if mrkrs[m_ix] is not None else ""
        _s = ln_sty[s_ix] if ln_sty[s_ix] is not None else ""
        _c = clrs[c_ix] if clrs[c_ix] is not None else ""
        print(_m,_s,_c)
        _fmt = ""+_m+_s+_c
        fmt.append(_fmt)
    width = np.random.uniform(600, 1200)
    height = np.random.uniform(450, 800)
    legend_loc = np.random.randint(0, 9)
    fontsize = np.random.randint(12,16)
    gid = np.random.randint(0,2)
    return fmt, legend_loc, width, height, fontsize, gid


    def matplotlib(self, val_data):
        fmt = self.get_plt_fmt(len(ln_data))

    def plotly(self): 

txt = '/home/sahmed9/Documents/data/charts/chartdata/texture/t1.png'
t_img = plt.imread(txt)

val_data = js_obj['task6']['output']['data series']

ct_id = []
ax_id = [] 
c_ax_title = []
text_roles = js_obj['task6']['input']['task3_output']['text_roles']
for r in text_roles :
    if r['role'] == 'axis_title' :
        ax_id.append(r['id'])
    if r['role'] == 'chart_title' :
        ct_id.append(r['id'])


min_x_axis = sorted(js_obj['task6']['input']['task4_output']['axes']['x-axis'], key= lambda k:k['tick_pt']['x'])[0]
max_x_axis = sorted(js_obj['task6']['input']['task4_output']['axes']['x-axis'], key= lambda k:k['tick_pt']['x'], reverse=True)[0]
min_y_axis = sorted(js_obj['task6']['input']['task4_output']['axes']['y-axis'], key= lambda k:k['tick_pt']['y'])[0]
max_y_axis = sorted(js_obj['task6']['input']['task4_output']['axes']['y-axis'], key= lambda k:k['tick_pt']['y'], reverse=True)[0]
print([min_x_axis['x'], max_x_axis['x'], min_y_axis['y'], max_y_axis['y']])

for _ in js_obj['task6']['input']['task2_output']['text_blocks'] :
    if _['id'] == min_x_axis['id'] :
        ex1 = float(_['text'])
    elif _['id'] == max_x_axis['id'] :
        ex2 = float(_['text'])
    elif _['id'] == min_y_axis['id'] :
        ex4 = float(_['text'])
    elif _['id'] == max_y_axis['id']:
        ex3 = float(_['text'])
    elif _['id'] in ct_id:
        c_tit_txt = _['text']
    elif _['id']  in ax_id:
        c_x = (_['polygon']['x0']+_['polygon']['x1']+_['polygon']['x2']+_['polygon']['x3'])/4
        c_y = (_['polygon']['y0']+_['polygon']['y1']+_['polygon']['y2']+_['polygon']['y3'])/4
        c_ax_title.append((_['text'],(c_x, c_y)))


print([ex1, ex2, ex3, ex4])

fm, legend_loc, width, height, fontsize, grid_ = get_plt_fmt(len(val_data))
# fig, ax = plt.subplots(figsize=(10,10))
fig, ax = plt.subplots(figsize=(6.95,4.25))
fig, ax = plt.subplots(figsize=(width/100,height/100))
# ax.imshow(t_img, extent=[ex1, ex2, ex3, ex4])
l = []
plt_info = []
for v_ix, v in enumerate(val_data) : 
    n = v['name']
    d = v['data']
    x_ =  [_['x'] for _ in d ]
    y_ =  [_['y'] for _ in d ]
    l.append(n)
    print(n)
    plt_info.append(ax.plot(x_, y_, fm[v_ix], label=n))

# legend_loc = 0 
if grid_ :
    ax.grid(which='major', linewidth=1.2)
    ax.grid(which='minor', linewidth=0.6)
    ax.grid(visible=True, which='both')
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.minorticks_on()
# ax.imshow(t_img, extent=[500, 800, 0, 900,])
plt.legend(l, loc=legend_loc, fontsize='xx-large')
plt.tight_layout()
# plt.legend(l)
sa = '/home/sahmed9/Documents/data/tash/cht_out/New Folder/Figure_1a.png'
plt.savefig(sa)
plt.show()

fls = os.listdir('/home/sahmed9/Documents/data/charts/release_ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/line_mask2')
text_class = {}
for f in fls : 
    js_obj = json.load(open(os.path.join(js_dir, f[:-4]+'.json')))
    text_roles = js_obj['task6']['input']['task3_output']['text_roles']
    for r in text_roles :
        # print(r)
        if r['role'] in text_class :
            text_class[r['role']]+=1
        else :
            text_class.update({r['role']:1})
        # if r['role'] in( 'tick_grouping', 'other', 'value_label', 'mark_label' ) :
        if r['role'] in( 'tick_grouping' ) :
            print('\n', r)
            print(f)


figw, figh = fig.get_size_inches()*fig.dpi ## width, height in pixels
ax_box = ax.get_position()
ax_box = ax.transAxes.transform([[0,0],[1,1]])

img__n =Image.new("RGB", (int(figw), int(figh)))
# im_ = ImageDraw.Draw(img__n)
im_ = ImageDraw.Draw(img__n)

# im_.rectangle([(ax_box.xmi * figw,ax_box.y0 * figh), (ax_box.x1 * figw, ax_box.y1 * figh)], fill='blue', width=1)
# im_.text((ax_box.x0*figw , ax_box.y0*figh),str(ax_box.x0*figw) +","+str(ax_box.y0 * figh) , fill=(255,255,255,255))
# im_.text((ax_box.x1*figw , ax_box.y1*figh),str(ax_box.x1*figw) +","+str(ax_box.y1 * figh) , fill=(255,255,255,255))

im_.rectangle([(ax_box[0][0], figh-ax_box[0][1]), (ax_box[1][0], figh-ax_box[1][1])], fill='blue', width=1)
im_.text((ax_box[0][0], figh-ax_box[0][1]), str(ax_box[0][0]) +","+str(figh-ax_box[0][1]) , fill=(255,255,255,255))
im_.text((ax_box[1][0], figh-ax_box[1][1]), str(ax_box[1][0]) +","+str(figh-ax_box[1][1]) , fill=(255,255,255,255))

for p in plt_info : 
    x, y = p[0].get_data() # returns task6b values -> convert to 6a 
    print(x,y)
    z2 = [(_x, _y) for _x, _y in zip(x,y)]
    disp = []
    for z2z2 in z2 :
        d1, d2 =  ax.transData.transform(z2z2)
        print(d1, d2)
        d2 = figh - d2
        print(d1, d2)
        disp.append((d1, d2))
    im_.line(disp , fill='green', width=1)
    # im_.line([(_y, _x) for _x, _y in zip(x,y)], fill='red', width=1)


img__n.show()



xm_t = ax.xaxis.get_major_ticks()


xtickslocs = ax.get_xticks()
ymin, _ = ax.get_ylim()
print('xticks pixel coordinates')
print(ax.transData.transform([(xtick, ymin) for xtick in xtickslocs]))
print('label bounding boxes')
print([l.get_window_extent() for l in ax.get_xticklabels()])