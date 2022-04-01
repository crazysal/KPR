f = '/home/sahmed9/reps/KeyPointRelations/cache/hms/13a/PMC548283___1471-2458-5-1-3.pkl'
sv_d = '/home/sahmed9/reps/KeyPointRelations/cache/ops/13a/'


f = '/home/sahmed9/reps/KeyPointRelations/cache/hms/13b/PMC548283___1471-2458-5-1-3.pkl'
sv_d = '/home/sahmed9/reps/KeyPointRelations/cache/ops/13b/'


f = '/home/sahmed9/reps/KeyPointRelations/cache/hms/13c/PMC548283___1471-2458-5-1-3.pkl'
sv_d = '/home/sahmed9/reps/KeyPointRelations/cache/ops/13c/'



f = '/home/sahmed9/reps/KeyPointRelations/cache/hms/13d/PMC548283___1471-2458-5-1-3.pkl'
sv_d = '/home/sahmed9/reps/KeyPointRelations/cache/ops/13d/'

f = '/home/sahmed9/reps/KeyPointRelations/cache/hms/13e/PMC548283___1471-2458-5-1-3.pkl'
sv_d = '/home/sahmed9/reps/KeyPointRelations/cache/ops/13e/'

f = '/home/sahmed9/reps/KeyPointRelations/cache/hms/13f/PMC548283___1471-2458-5-1-3.pkl'
sv_d = '/home/sahmed9/reps/KeyPointRelations/cache/ops/13f/'

f = '/home/sahmed9/reps/KeyPointRelations/cache/hms/13g/PMC548283___1471-2458-5-1-3.pkl'
sv_d = '13g'

import pickle 
import os.path as osp 
from matplotlib import pyplot as plt

f_d = '/home/sahmed9/reps/KeyPointRelations/cache/hms/'
sv_ = '/home/sahmed9/reps/KeyPointRelations/cache/ops/'

fold = ['13a', '13b', '13c', '13d','13e', '13f', '13g']

im = ['PMC548283___1471-2458-5-1-3.pkl','PMC1166547___1471-2156-6-26-4.pkl','PMC1166548___1471-2156-6-29-4.pkl','PMC1193525___pgen.0010015.g005.pkl','PMC1200427___pgen.0010032.g002.pkl','PMC1310579___pgen.0010070.g001.pkl','PMC1403762___1471-2458-6-16-1.pkl','PMC1866731___1471-2156-6-S1-S52-1.pkl','PMC2263030___1471-2458-8-60-2.pkl','PMC2292166___1471-2458-8-80-1.pkl','PMC2464733___pgen.1000137.g012.pkl','PMC2486442___ddn14604.pkl','PMC2672343___ijerph-06-00232f3.pkl','PMC2672397___ijerph-06-01145f1.pkl','PMC2681204___ijerph-06-01298f1.pkl']



for i in im : 
    for fld in fold :
        fl = osp.join(f_d, fld, i)
        print('File:', fl)
        fl = open(fl, 'rb')
        fl = pickle.load(fl)
        plt.figure()
        fig, axs = plt.subplots(2, 4)
        axs[0,0].imshow(fl[0, :, :])
        axs[0,1].imshow(fl[1, :, :])
        axs[0,2].imshow(fl[2, :, :])
        axs[0,3].imshow(fl[3, :, :])
        axs[1,0].imshow(fl[4, :, :])
        axs[1,1].imshow(fl[5, :, :])
        axs[1,2 ].imshow(fl[6, :, :])
        axs[1,3].imshow(fl[7, :, :])
        fig.savefig(osp.join(sv_, fld ,i[:-4]+'.png'))
        plt.clf()



