import argparse
import os 

class Configs():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Keypoint Relations')
        parser.add_argument('--conf_train', type=str, default='9a')
        parser.add_argument('--conf_test', type=str, default='0t')
        parser.add_argument('--conf_val', type=str, default='0v')
        # Model run    
        parser.add_argument('--loss',         type=str,  help='mse/ce')
        parser.add_argument('--model',        type=str,  help='basic/basicPre/hgNet')
        parser.add_argument('--in_chanl',     type=int,  help=' Input Channels')
        parser.add_argument('--in_sz',        type=int,  help='Input Size')
        parser.add_argument('--targ_sz',      type=int,  help='Target Size')
        parser.add_argument('--t_batch_size', type=int,  help='Train Bach Size')
        parser.add_argument('--v_batch_size', type=int,  help='Val Bach Size')
        parser.add_argument('--split',        type=str,  help='train/val/test')
        parser.add_argument('--epoch',        type=int,  help='Total epochs')
        # Files                    
        parser.add_argument('--rd',           type=str,  help='Project Root directory')
        parser.add_argument('--img_d',        type=str,  help='Image directory')
        parser.add_argument('--cache_dir',    type=str,  help='Cache directory')
        parser.add_argument('--cache_file',   type=str,  help='Model Cache File')
        parser.add_argument('--resume',       type=str,  help='Model checkpoint File')
        parser.add_argument('--test',         type=str,  help='Test folder directory')
        parser.add_argument('--op_dir',       type=str,  help='Test output directory')
        # Misc
        parser.add_argument('--val_ep',       type=int,  help='Validation Epoch Frequency')
        parser.add_argument('--lr',           type=float,help='Learning Rate')
        parser.add_argument('--cuda',         type=bool, help='GPU avlbl')
        parser.add_argument('--saver',        type=int,  help='Save Frequency')                            
        parser.add_argument('--scheduler',    type=int,  help='Learning Rate scheduler')

        self.parser = parser
                                    

    def getConfig(self, cnfg):
        opt = self.parser.parse_args()

        if cnfg == '9' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = True
            opt.model_mode  = 'M1'
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 24
            
            opt.split = "train"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_9'
            opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_9/_ep_19_.pth'
            opt.epoch = 20
            opt.saver = 1
            opt.val_ep = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '9v' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            # opt.batch_size = opt.v_batch_size =[6,6,6,6]
            opt.batch_size = opt.v_batch_size = 24
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.model_mode  = 'M1'
            opt.isDP = True

            opt.split = "val"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_9'

            opt.epoch = 20
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '9t' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.batch_size = 1
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.model_mode  = 'M1'
            opt.isDP = False

            opt.split = "test"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images/test_images_pmc_line/'
            # opt.img_d = 'data/images/test_images_dave/'
            # opt.img_d = 'data/images/scatter/'
            # opt.img_d = 'data/images/scatter2/'
            # opt.img_d = 'data/images/test_pmc_line_clean/'
            # opt.js_d = 'data/JSONs'
            # opt.img_d = 'data/images/test_pmc_line_clean/'
            opt.js_d = 'data/JSONs'

            # opt.t_js_d = 'data/line_json_test'
            # opt.t_js_d = 'data/dave_line_json_test'
            # opt.t_js_d = 'data/JSONs/scatter'
            # opt.t_js_d = 'data/JSONs/scatter2'
            # opt.t_js_d = 'data/dave_line_json_test'
            opt.t_js_d = 'data/line_json_test_clean'
            
            opt.fl_pmc = 'data/fl_figseer.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_9trn'
            opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_9/_ep_19_.pth'

            # opt.op_dir='/home/sahmed9/reps/KeyPointRelations/cache/hms/9'
            # opt.op_dir='/home/sahmed9/reps/KeyPointRelations/cache/hms/9dave'
            # opt.op_dir='/home/sahmed9/reps/KeyPointRelations/cache/hms/9_scatter'
            # opt.op_dir='/home/sahmed9/reps/KeyPointRelations/cache/hms/9_clean_line'
            opt.op_dir='/home/sahmed9/reps/KeyPointRelations/cache/hms/9_clean_train'

            opt.epoch = 15
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '9a' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = True
            opt.model_mode  = 'M2'
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 5
            
            opt.split = "train"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_9a'
            opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_9/_ep_19_.pth'
            opt.epoch = 20
            opt.saver = 1
            opt.val_ep = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '9av' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            # opt.batch_size = opt.v_batch_size =[6,6,6,6]
            opt.batch_size = opt.v_batch_size = 50
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.model_mode  = 'M2'
            opt.isDP = True

            opt.split = "val"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_9a'

            opt.epoch = 20
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '9at' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.batch_size = 1
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.model_mode  = 'M2'
            opt.isDP = True

            opt.split = "test"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_9a'

            # opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_9/_ep_19_.pth'
            opt.op_dir='/home/sahmed9/reps/KeyPointRelations/cache/hms/9a'

            opt.epoch = 15
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '10' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            opt.model_mode  = 'M1M2'
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 6
            
            opt.split = "train"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_10'
            # opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_9/_ep_19_.pth'
            opt.epoch = 20
            opt.saver = 1
            opt.val_ep = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '10v' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            # opt.batch_size = opt.v_batch_size =[6,6,6,6]
            opt.batch_size = opt.v_batch_size = 6
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.model_mode  = 'M1M2'
            opt.isDP = True

            opt.split = "val"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_10'

            opt.epoch = 20
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '10t' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.batch_size = 1
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.model_mode  = 'M1M2'
            opt.isDP = True

            opt.split = "test"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_10'

            # opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_9/_ep_19_.pth'
            opt.op_dir='/home/sahmed9/reps/KeyPointRelations/cache/hms/10'

            opt.epoch = 15
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True


        if cnfg == '10a' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            opt.model_mode  = 'M1M2'
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 6
            
            opt.split = "train"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_10a'
            # opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_9/_ep_19_.pth'
            opt.epoch = 20
            opt.saver = 1
            opt.val_ep = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '10av' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            # opt.batch_size = opt.v_batch_size =[6,6,6,6]
            opt.batch_size = opt.v_batch_size = 6
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.model_mode  = 'M1M2'
            opt.isDP = True

            opt.split = "val"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_10a'

            opt.epoch = 20
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '10at' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.batch_size = 1
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.model_mode  = 'M1M2'
            opt.isDP = True

            opt.split = "test"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_10'

            # opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_9/_ep_19_.pth'
            opt.op_dir='/home/sahmed9/reps/KeyPointRelations/cache/hms/10'

            opt.epoch = 15
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True


        if cnfg == '10b' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            opt.model_mode  = 'M1M2'
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 6
            
            opt.split = "train"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_10b'
            # opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_9/_ep_19_.pth'
            opt.epoch = 20
            opt.saver = 1
            opt.val_ep = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '10bv' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            # opt.batch_size = opt.v_batch_size =[6,6,6,6]
            opt.batch_size = opt.v_batch_size = 2
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.model_mode  = 'M1M2'
            opt.isDP = True

            opt.split = "val"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_10b'

            opt.epoch = 20
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '10bt' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.batch_size = 1
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.model_mode  = 'M1M2'
            opt.isDP = False

            opt.split = "test"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images/test_images_pmc_line'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'

            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_10b'

            opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_10b/_ep_2_.pth'
            
            opt.op_dir='/home/sahmed9/reps/KeyPointRelations/cache/hms/10b'

            opt.epoch = 15
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True
        if cnfg == '11' :            
            opt.model_mode  = 'PF'
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 12
            
            opt.split = "train"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_11'
            opt.load_file = None
            opt.epoch = 20
            opt.saver = 1
            opt.val_ep = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '11v' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            # opt.batch_size = opt.v_batch_size =[6,6,6,6]
            opt.batch_size = opt.v_batch_size = 4
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.model_mode  = 'M1M2'
            opt.isDP = True

            opt.split = "val"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_11'

            opt.epoch = 20
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '11a' :            
            opt.model_mode  = 'PF'
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 6
            
            opt.split = "train"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_11a'
            # opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_11/_ep_19_.pth'
            opt.load_file = None
            opt.epoch = 20
            opt.saver = 1
            opt.val_ep = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '11av' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            # opt.batch_size = opt.v_batch_size =[6,6,6,6]
            opt.batch_size = opt.v_batch_size = 2
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.model_mode  = 'M1M2'
            opt.isDP = True

            opt.split = "val"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_11a'

            opt.epoch = 20
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '11aa' :            
            opt.model_mode  = 'PF'
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 8
            
            opt.split = "train"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_11aa'
            opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_11a/_ep_10_.pth'
            opt.epoch = 20
            opt.saver = 1
            opt.val_ep = 1
            opt.lr = 1e-4
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '11aav' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            # opt.batch_size = opt.v_batch_size =[6,6,6,6]
            opt.batch_size = opt.v_batch_size = 2
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.model_mode  = 'M1M2'
            opt.isDP = True

            opt.split = "val"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_11aa'

            opt.epoch = 20
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '11b' :            
            opt.model_mode  = 'PF'
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 8
            
            opt.split = "train"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_11b'
            opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_11og/_ep_2_.pth'
            opt.epoch = 20
            opt.saver = 1
            opt.val_ep = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '11bv' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            # opt.batch_size = opt.v_batch_size =[6,6,6,6]
            opt.batch_size = opt.v_batch_size = 2
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.model_mode  = 'M1M2'
            opt.isDP = True

            opt.split = "val"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_11b'

            opt.epoch = 20
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True


        if cnfg == '11t' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.batch_size = 1
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.model_mode  = 'M1M2'
            opt.isDP = False

            opt.split = "test"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images/test_images_pmc_line'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'

            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_11'

            # opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_10b/_ep_2_.pth'
            opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_11/_ep_19_.pth'
            
            opt.op_dir='/home/sahmed9/reps/KeyPointRelations/cache/hms/11'

            opt.epoch = 15
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '11at' :            
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.batch_size = 1
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.model_mode  = 'M1M2'
            opt.isDP = False

            opt.split = "test"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images/test_images_pmc_line'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'

            opt.fl_pmc = 'data/split4_train_task6_all.pkl'
            opt.fl_synth = 'data/train_task6_syntyPMC.pkl'
            opt.fl_fs = 'data/fl_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_11'

            opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_11a/_ep_19_.pth'
            
            opt.op_dir='/home/sahmed9/reps/KeyPointRelations/cache/hms/11a'

            opt.epoch = 15
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '12' :            
            opt.dataset = 'line sequence'
            opt.model_mode  = 'PF'
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 12
            
            opt.split = "train"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_12'
            # opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_11/_ep_19_.pth'
            opt.load_file = None
            opt.epoch = 50
            opt.saver = 1
            opt.val_ep = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '12v' :      
            opt.dataset = 'line sequence'
            opt.model_mode  = 'PF'         
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            # opt.batch_size = opt.v_batch_size =[6,6,6,6]
            opt.batch_size = opt.v_batch_size = 2
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = True

            opt.split = "val"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_12'

            opt.epoch = 20
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '13at' :            
            opt.dataset = 'line sequence'
            opt.model_mode  = 'mpn'
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 6
            
            opt.split = "test"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images/test_images_pmc_line'
            opt.img_d = 'data/images/test_pmc_line_clean'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'

            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_13at'
            opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_13a/_ep_40_.pth'
            # opt.load_file = None
            opt.op_dir='/home/sahmed9/reps/KeyPointRelations/cache/hms/13a'

            opt.epoch = 50
            opt.saver = 10
            opt.val_ep = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True
        
        if cnfg == '13a' :            
            opt.dataset = 'line sequence'
            opt.model_mode  = 'mpn'
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 6
            
            opt.split = "train"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_13a'
            # opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_11/_ep_19_.pth'
            opt.load_file = None
            opt.epoch = 50
            opt.saver = 10
            opt.val_ep = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '13av' :      
            opt.dataset = 'line sequence'
            opt.model_mode  = 'mpn'         
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            # opt.batch_size = opt.v_batch_size =[6,6,6,6]
            opt.batch_size = opt.v_batch_size = 2
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = True

            opt.split = "val"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_13a'

            opt.epoch = 50
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '13bt' :            
            opt.dataset = 'line sequence'
            opt.model_mode  = 'spn'
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 6
            
            opt.split = "test"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images/test_images_pmc_line'
            opt.img_d = 'data/images/test_pmc_line_clean'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'

            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_13bt'
            opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_13b/_ep_40_.pth'
            # opt.load_file = None
            opt.op_dir='/home/sahmed9/reps/KeyPointRelations/cache/hms/13b'

            opt.epoch = 50
            opt.saver = 10
            opt.val_ep = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True
        if cnfg == '13b' :            
            opt.dataset = 'line sequence'
            opt.model_mode  = 'spn'
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 6
            
            opt.split = "train"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_13b'
            # opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_11/_ep_19_.pth'
            opt.load_file = None
            opt.epoch = 50
            opt.saver = 10
            opt.val_ep = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '13bv' :      
            opt.dataset = 'line sequence'
            opt.model_mode  = 'spn'         
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            # opt.batch_size = opt.v_batch_size =[6,6,6,6]
            opt.batch_size = opt.v_batch_size = 2
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = True

            opt.split = "val"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_13b'

            opt.epoch = 50
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '13c' :            
            opt.dataset = 'line sequence'
            opt.model_mode  = 'cpn'
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 6
            
            opt.split = "train"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_13c'
            # opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_11/_ep_19_.pth'
            opt.load_file = None
            opt.epoch = 50
            opt.saver = 10
            opt.val_ep = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '13cv' :      
            opt.dataset = 'line sequence'
            opt.model_mode  = 'cpn'         
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            # opt.batch_size = opt.v_batch_size =[6,6,6,6]
            opt.batch_size = opt.v_batch_size = 2
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = True

            opt.split = "val"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_13c'

            opt.epoch = 50
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '13ct' :            
            opt.dataset = 'line sequence'
            opt.model_mode  = 'cpn'
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 6
            
            opt.split = "test"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images/test_images_pmc_line'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'

            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            
            opt.cache_dir = 'cache'
            opt.cache_file = 'AllPMC_PointDetectorHgNet_13ct'
            opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_13c/_ep_40_.pth'
            # opt.load_file = None
            opt.op_dir='/home/sahmed9/reps/KeyPointRelations/cache/hms/13c'

            opt.epoch = 50
            opt.saver = 10
            opt.val_ep = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        
        if cnfg == '13gt' :            
            opt.dataset = 'line sequence'
            opt.model_mode  = 'nopn'
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 6
            
            opt.split = "test"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images/test_images_pmc_line'
            opt.img_d = 'data/images/test_pmc_line_clean'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'

            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            
            opt.cache_dir = 'cache'
            opt.cache_file = 'lineseq_fcn50_13g'
            opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/lineseq_fcn50_13g/_ep_4_.pth'
            # opt.load_file = None
            opt.op_dir='/home/sahmed9/reps/KeyPointRelations/cache/hms/13g'

            opt.epoch = 50
            opt.saver = 10
            opt.val_ep = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '13g' :            
            opt.dataset = 'line sequence'
            opt.model_mode  = 'nopn'
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 6
            
            opt.split = "train"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'lineseq_fcn50_13g'
            # opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_11/_ep_19_.pth'
            opt.load_file = None
            opt.epoch = 5
            opt.saver = 1
            opt.val_ep = 1
            opt.lr = 1e-4
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '13gv' :      
            opt.dataset = 'line sequence'
            opt.model_mode  = 'nopn'         
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            # opt.batch_size = opt.v_batch_size =[6,6,6,6]
            opt.batch_size = opt.v_batch_size = 2
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = True

            opt.split = "val"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'lineseq_fcn50_13g'

            opt.epoch = 5
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-4
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '13dt' :            
            opt.dataset = 'line sequence'
            opt.model_mode  = 'mpn'
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 6
            
            opt.split = "test"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images/test_images_pmc_line'
            opt.img_d = 'data/images/test_pmc_line_clean'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'

            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            
            opt.cache_dir = 'cache'
            opt.cache_file = 'lineseq_fcn50_13d'
            opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/lineseq_fcn50_13d/_ep_4_.pth'
            # opt.load_file = None
            opt.op_dir='/home/sahmed9/reps/KeyPointRelations/cache/hms/13d'

            opt.epoch = 50
            opt.saver = 10
            opt.val_ep = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '13d' :            
            opt.dataset = 'line sequence'
            opt.model_mode  = 'mpn'
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 6
            
            opt.split = "train"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'lineseq_fcn50_13d'
            # opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_11/_ep_19_.pth'
            opt.load_file = None
            opt.epoch = 5
            opt.saver = 1
            opt.val_ep = 1
            opt.lr = 1e-4
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '13dv' :      
            opt.dataset = 'line sequence'
            opt.model_mode  = 'mpn'         
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            # opt.batch_size = opt.v_batch_size =[6,6,6,6]
            opt.batch_size = opt.v_batch_size = 2
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = True

            opt.split = "val"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'lineseq_fcn50_13d'

            opt.epoch = 5
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-4
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '13et' :            
            opt.dataset = 'line sequence'
            opt.model_mode  = 'spn'
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 6
            
            opt.split = "test"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images/test_images_pmc_line'
            opt.img_d = 'data/images/test_pmc_line_clean'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'

            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            
            opt.cache_dir = 'cache'
            opt.cache_file = 'lineseq_fcn50_13e'
            opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/lineseq_fcn50_13e/_ep_4_.pth'
            # opt.load_file = None
            opt.op_dir='/home/sahmed9/reps/KeyPointRelations/cache/hms/13e'

            opt.epoch = 50
            opt.saver = 10
            opt.val_ep = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True
        if cnfg == '13e' :            
            opt.dataset = 'line sequence'
            opt.model_mode  = 'spn'
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 6
            
            opt.split = "train"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'lineseq_fcn50_13e'
            # opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_11/_ep_19_.pth'
            opt.load_file = None
            opt.epoch = 5
            opt.saver = 1
            opt.val_ep = 1
            opt.lr = 1e-4
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '13ev' :      
            opt.dataset = 'line sequence'
            opt.model_mode  = 'spn'         
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            # opt.batch_size = opt.v_batch_size =[6,6,6,6]
            opt.batch_size = opt.v_batch_size = 2
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = True

            opt.split = "val"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'lineseq_fcn50_13e'

            opt.epoch = 5
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-4
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '13ft' :            
            opt.dataset = 'line sequence'
            opt.model_mode  = 'cpn'
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 6
            
            opt.split = "test"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images/test_pmc_line_clean'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'

            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            
            opt.cache_dir = 'cache'
            opt.cache_file = 'lineseq_fcn50_13f'

            opt.load_file ='cache/saves/lineseq_fcn50_13f/_ep_4_.pth'
            # opt.load_file = None
            opt.op_dir='cache/hms/13f'

            opt.epoch = 50
            opt.saver = 10
            opt.val_ep = 1
            opt.lr = 1e-3
            opt.scheduler = 1
            opt.cuda = True
        if cnfg == '13f' :            
            opt.dataset = 'line sequence'
            opt.model_mode  = 'cpn'
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = False
            # opt.batch_size = opt.t_batch_size = [6,6,6,6]
            opt.batch_size = opt.t_batch_size = 6
            
            opt.split = "train"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'lineseq_fcn50_13f'
            # opt.load_file ='/home/sahmed9/reps/KeyPointRelations/cache/saves/AllPMC_PointDetectorHgNet_11/_ep_19_.pth'
            opt.load_file = None
            opt.epoch = 5
            opt.saver = 1
            opt.val_ep = 1
            opt.lr = 1e-4
            opt.scheduler = 1
            opt.cuda = True

        if cnfg == '13fv' :      
            opt.dataset = 'line sequence'
            opt.model_mode  = 'cpn'         
            opt.in_chanl = 3
            opt.in_sz = 512
            opt.targ_sz = 128
            # opt.batch_size = opt.v_batch_size =[6,6,6,6]
            opt.batch_size = opt.v_batch_size = 2
            opt.stacks = 3
            opt.blocks = 2
            opt.classes = 5
            opt.isDP = True

            opt.split = "val"            
            opt.rd = '/home/sahmed9/reps/KeyPointRelations'
            opt.img_d = 'data/images'
            opt.js_d = 'data/JSONs'
            opt.t_js_d = 'data/line_json_test'
            opt.fl_pmc = 'data/split4_train_task6_line.pkl'
            opt.fl_synth = 'data/split4_train_task6_line2.pkl'
            opt.fl_fs = 'data/fl_gt1_figseer.pkl'
            opt.cache_dir = 'cache'
            opt.cache_file = 'lineseq_fcn50_13f'

            opt.epoch = 5
            opt.val_ep = 1
            opt.saver = 1
            opt.lr = 1e-4
            opt.scheduler = 1
            opt.cuda = True
        return opt 