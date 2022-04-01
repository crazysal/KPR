
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import copy 

from model.hg_net import HourglassNet, Bottleneck
class PtFormer(nn.Module):
    ''''''
    def __init__(self, **kwd):
        super(PtFormer, self).__init__()
        config = kwd['config']
        # print('init ptformer')
        # print(config)
        self.max_points = 40
        self.max_lines = 8
        self.d_model = 256
        # self.ntoken = 128*128*10+3
        # self.start_token = torch.tensor(128*128*10+1)
        # self.stop_token = torch.tensor(128*128*10+2)
        # self.pad_token = torch.tensor(128*128*10+3)
        self.ntoken = 128*128+2
        self.start_token = torch.tensor(16384).cuda()
        self.stop_token = torch.tensor(16385).cuda()
        # self.pad_token = torch.tensor(128*128*10+3)
        chnl_ = config.heatmap_chan = 256
        # self.hg = HourglassNet(Bottleneck, num_stacks=config.stacks, num_blocks=config.blocks, num_classes=config.classes)
        self.hg = HourglassNet(Bottleneck, num_stacks=1, num_blocks=2, num_classes=2)
        # # encoder_layer = nn.TransformerEncoderLayer(d_model=config.enc_dim, nhead=config.enc_head)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        # # decoder_layer = nn.TransformerDecoderLayer(d_model=config.dec_dim, nhead=config.dec_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=4)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.e_lookup = nn.Embedding(self.ntoken, self.d_model)
        self.project = nn.Linear(self.d_model, self.ntoken)
        self.project_coord = nn.Linear(self.d_model, 2)
        self.coord_range = {}
        key = 0
        for x_ in range(128) :
            for y_ in range(128) :
                self.coord_range.update({(x_, y_):key})
                key+=1
        print('self.coord_range', self.coord_range)
        # print(self.e_lookup)
        self.coord_range_rev = {}
        key = 0
        for x_ in range(128) :
            for y_ in range(128) :
                self.coord_range_rev.update({key: (x_, y_)})
                key+=1
        self.coord_range_rev.update({16384: 'start'})
        self.coord_range_rev.update({16385: 'stop'})
        print('self.coord_range_rev', self.coord_range_rev)
        # exit()
        # self.ofx_range = {0 :(0, 0), 1 :(0, 0.2), 2 :(0.2, 0.4), 3 :(0.4, 0.6), 4 :(0.6, 0.8), 5 :(0.8, 1) }
        # self.ofy_range = {0 :(0, 0), 1 :(0, 0.2), 2 :(0.2, 0.4), 3 :(0.4, 0.6), 4 :(0.6, 1)}
    def forward(self, x, edge_list, point_list_, point_norm_list, split, _, sz):
        # print(x.shape)
        if sz is not None : 
            pnl = []
            point_list, pt_idx = point_list_ 
            for idx, osz in enumerate(sz) : 
                ln_ = []
                points = point_list[idx]
                for pt in points :
                    pt0, pt1 = pt
                    pt0/=osz[0]
                    pt1/=osz[1]
                    ln_.append((pt0, pt1))
                pnl.append(ln_)
            # print(pnl)
            point_norm_list = pnl
        decoder_loss = _
        # print('1', torch.cuda.memory_allocated()*100/torch.cuda.max_memory_allocated())
        x, x_ = self.hg(x)
        # print('2', torch.cuda.memory_allocated()*100/torch.cuda.max_memory_allocated())
        # print('from bbone', len(x))
        # for _, __ in zip(x, x_) : 
            # print(_.shape, __.shape)
        if decoder_loss :
            seq, src_mask, pad_mask = self.get_encoder_sequence(x, x_, point_list_, point_norm_list, sz)
            # print('3', torch.cuda.memory_allocated()*100/torch.cuda.max_memory_allocated())
            # print(seq)
            memory = self.encode_lines(seq, src_mask, pad_mask)
            # print('4', torch.cuda.memory_allocated()*100/torch.cuda.max_memory_allocated())
            # print('memory', memory.shape)
            deq, tgt_mask, tpad_mask, labels = self.get_decoder_sequence(point_list_, sz)
            # print('5', torch.cuda.memory_allocated()*100/torch.cuda.max_memory_allocated())
            # print('deq start', deq.shape)
            # print(deq)
            if split !='test' : 

                output = self.decode_lines(deq[:-1, :, :], memory, tgt_mask[:-1, :-1], tpad_mask[:, :-1])
                # print('6', torch.cuda.memory_allocated()*100/torch.cuda.max_memory_allocated())
                # print('final model op', output.shape)
                op_deq = self.project(output)
                op_coord = self.project_coord(output)
                # print('7', torch.cuda.memory_allocated()*100/torch.cuda.max_memory_allocated())
                return {'hm': x, 'deq' :op_deq, 'cor': op_coord , 'pnl': (point_norm_list, point_list_[1]), 'type' : 'transformer', 'decoder_loss':decoder_loss, 'dec_label':labels}
            else : 
                output=  self.decode_lines_pointwise(deq, memory, tgt_mask, tpad_mask)
                # print('output', output)
                # point_list, pt_idx = point_list_
                # # print('point_list, pt_idx', point_list, pt_idx)

                # point_list, pt_idx  = point_list[0], pt_idx[0] 
                # point_list = torch.tensor(point_list).split(pt_idx)
                # for l, p in zip(output, point_list) :
                #     print('pred :', l)
                #     print('gt:', p)
                #     print('\n')
                exit()
                return output
        else :
            # print('8', torch.cuda.memory_allocated()*100/torch.cuda.max_memory_allocated())
            return {'hm': x, 'type' : 'transformer', 'decoder_loss':decoder_loss}

    def get_decoder_sequence(self, point_list_, og_size=None):
        point_list, pt_idx = point_list_ 
        pos_embed, labla = self.get_pos_embedding(point_list, og_size) ## N x model_dim  where N = (total lines in batch)
        # print('labla from gt', len(labla))
        # for l in labla :
        #     print(l.shape, l)
        feats = []
        masks = []
        labels = [] 
        for p_ix, pos_ in enumerate(pos_embed) :
            pl = point_list[p_ix]

            point_count = pt_idx[p_ix]
            pos_  = pos_.split(point_count)
            labla_  = labla[p_ix].split(point_count)
            # print('labla_, after split', labla_)
            nop   = []
            mask  = []
            label = [] 
            for o_idx, o in enumerate(pos_) : 
                s = self.start_token.unsqueeze(0)
                e = self.stop_token.unsqueeze(0)
                # print('start, end',s,  s.shape, e, e.shape)
                if o.shape[0] < self.max_points :  ## add padding for lines and mask 
                    # print('label shape/max pt', o.shape, self.max_points)
                    pad = torch.zeros(self.max_points - o.shape[0], o.shape[1]).cuda()
                    nop.append(torch.cat((self.e_lookup(s), o, self.e_lookup(e), pad)))
                    # print('nop', nop[-1])
                    pad_id = [16386]*(self.max_points - o.shape[0])
                    # print('pad_id, len()', len(pad_id))
                    pad_id = torch.tensor(pad_id).long().cuda()
                    # print('pad_id, shape', pad_id.shape)
                    # label.append(torch.cat((labla_[o_idx],pad_id))) ## 16386 be pad ID 
                    # l_ = torch.cat((s,labla_[o_idx],e, pad_id))
                    l_ = torch.cat((labla_[o_idx],e, pad_id))
                    # print('catted, s+lbl+e+pad', l_.shape, l_)
                    label.append(l_) ## 16386 be pad ID 
                    m = torch.zeros(self.max_points+2).cuda()
                    # print('dec mask', m.shape, m)
                    m[o.shape[0]-(self.max_points):]+=1
                    # print('dec mask+1', m.shape, m)
                    mask.append(m.bool())
                elif o.shape[0] >= self.max_points :
                    # nop.append(o[:self.max_points, :])
                    nop.append(torch.cat((self.e_lookup(s),o[:self.max_points, :],self.e_lookup(e))))
                    m = torch.zeros(self.max_points+2).bool().cuda()
                    mask.append(m)
                    # l_ = torch.cat((s,labla_[o_idx][:self.max_points],e))
                    l_ = torch.cat((labla_[o_idx][:self.max_points],e))
                    label.append(l_) 

            # print('labels', len(label))
            nop = torch.stack(nop)
            mask = torch.stack(mask)
            label = torch.stack(label)
            # print('label', label.shape)
            # s = self.e_lookup(self.start_token).unsqueeze(0).unsqueeze(0).repeat(nop.shape[0], 1, 1)
            # e = self.e_lookup(self.stop_token).unsqueeze(0).unsqueeze(0).repeat(nop.shape[0], 1, 1)
            # ft = torch.cat((s,nop,e), dim=1)
            # mask = torch.cat((torch.zeros(s.shape[0],s.shape[1]).bool().cuda(), mask , torch.zeros(s.shape[0],s.shape[1]).bool().cuda()), dim=1)
            # s_ = torch.tensor([16384]).long().cuda().unsqueeze(0).repeat(label.shape[0],1)
            # e_ = torch.tensor([16385]).long().cuda().unsqueeze(0).repeat(label.shape[0],1)
            # label = torch.cat((s_, label , e_), dim=1)
            
            # print('lbl shape', label.shape)
             
            # feats.append(ft)
            feats.append(nop)
            masks.append(mask)
            labels.append(label)
         
        feats = torch.cat(feats, dim=0).permute(1,0,2)
        pad_masks = torch.cat(masks, dim=0)
        tgt_masks = torch.zeros((feats.shape[0], feats.shape[0])).bool().cuda()
        labels = torch.cat(labels)
        # print('decoder feats masks lbl returned', feats.shape, tgt_masks.shape, pad_masks.shape, labels.shape)
        return feats , tgt_masks, pad_masks, labels

    def get_encoder_sequence(self, x, x_, point_list_, point_raw_list, og_size=None) :
    
        ## 3 stacks, number output class == 5 
        ## x_ ->> list of outputss channel 256 -> this will give feature embed
        # hm1, hm2, hm3 = x # list of outputss channel 5  -> this will intensity 
        # intensity = torch.sigmoid(heatmaps_pred.squeeze())

        ## get features of respecive points from last stack 
        maps = x_[-1]
        # print('After Backbone', maps.shape)
        point_list, pt_idx = point_list_ 
        print('total point list, pnl', len(point_list),len(point_raw_list))
        tp = 0
        for p_, p__ in zip(point_list, point_raw_list) :
            print('total pts short, raw',len(p_), len(p__))
            tp+=len(p_)
        print('Total Points', tp)
        # print('total pt_idx', len(pt_idx))
        # for p_ in pt_idx :
        #     print(len(p_), p_)

        pos_embed, _ = self.get_pos_embedding(point_list, og_size) ## N x model_dim  where N = (total lines in batch)
        # print('pos_embed len', len(pos_embed))
        feats = []
        masks = [] 
        ## image (batch)-> lines(one seq) -> points(each token) 
        for image in range(maps.shape[0]) :
            op = []
            pl = point_list[image]
            pos_ = pos_embed[image]
            # print('got pos embedding', pos_.shape)
            point_count = pt_idx[image] 
            # print('got point_count', point_count)
            for pt_  in pl :
                lines = []
                for pt_ct in point_count :
                    pt0, pt1 = pt_[0], pt_[1]
                    if og_size is not None :
                        osz = og_size[image]
                        pt0/=osz[0]
                        pt0*=128
                        pt1/=osz[1]
                        pt1*=128 
                        if pt0 >=128 :
                            print('::: ERROR ::: x', pt0, osz) 
                            pt0 = 127
                        if pt1>=128 :
                            print('::: ERROR ::: y', pt1, osz) 
                            pt1 = 127
                    fp = maps[image, :, int(pt0), int(pt1) ]
                op.append(fp)
            # print('got feats', len(op), 'point_count', point_count)
            op = torch.stack(op)
            # print('got feats', op.shape)
            op = op.split(point_count)
            # print('got split feats', len(op))
            pos_ = pos_.split(point_count)
            # print('got splt pos', len(pos_))
            nop  = []
            mask = []
            for o_idx, o in enumerate(op) : 
                # print('each line feat', o.shape)
                s = self.e_lookup(self.start_token).unsqueeze(0)
                e = self.e_lookup(self.stop_token).unsqueeze(0)
                if o.shape[0] < self.max_points :  ## add padding for lines and mask 
                    # print('pad feats', self.max_points - o.shape[0], o.shape[1])
                    pad = torch.zeros(self.max_points - o.shape[0], o.shape[1]).cuda()
                    # print('o', o, o.shape)
                    # print('pos_[o_idx]', pos_[o_idx], pos_[o_idx].shape)
                    nop_ = o + pos_[o_idx]
                    # print('nop_', nop_, nop_.shape)
                    nop.append(torch.cat((s, nop_, e, pad)))
                    # print('nop[-1]', nop[-1], nop[-1].shape)
                    m = torch.zeros(self.max_points+2).cuda()
                    m[o.shape[0]-(self.max_points):]+=1
                    mask.append(m.bool())
                    # print('oshap, mask smol',o.shape[0], m, m.shape)
                elif o.shape[0] >= self.max_points :
                    # print('VERY LONG LINE SLICED')
                    nop_ = o[:self.max_points, :] + pos_[o_idx][:self.max_points, :]
                    nop.append(torch.cat((s, nop_, e)))
                    # print('nop[-1] big', nop[-1], nop[-1].shape)
                    m = torch.zeros(self.max_points+2).bool().cuda()
                    # print('mask big', m, m.shape)
                    mask.append(m)

            # print('Final len nop', len(nop), len(mask))
            nop = torch.stack(nop)
            mask = torch.stack(mask)
            # print('nop, padmask stacked', nop.shape, mask.shape)
            # print(nop)
            # print(mask)
            # nop += pos_
            # print('nop', nop.shape)
            # s = self.e_lookup(self.start_token).unsqueeze(0).unsqueeze(0).repeat(nop.shape[0], 1, 1)
            # print('start', s.shape)
            # e = self.e_lookup(self.stop_token).unsqueeze(0).unsqueeze(0).repeat(nop.shape[0], 1, 1)
            # print('end', e.shape)
            # ft = torch.cat((s,nop,e), dim=1)
            # mask = torch.cat((torch.zeros(s.shape[0],s.shape[1]).bool().cuda(), mask , torch.zeros(s.shape[0],s.shape[1]).bool().cuda()), dim=1)
            # print('final ft, pad mask', ft.shape, mask.shape)
            # print(ft)
            # feats.append(ft)
            feats.append(nop)
            masks.append(mask)
        ## final shape : N, S, E 
        
        feats = torch.cat(feats, dim=0).permute(1,0,2)
        pad_masks = torch.cat(masks, dim=0)
        src_masks = torch.zeros((feats.shape[0], feats.shape[0])).bool().cuda()
        # print('encoder feats masks returned', feats.shape, src_masks.shape, pad_masks.shape)
        # print(pad_masks)
        # print(src_masks)
        return feats , src_masks, pad_masks

    ## returns list of tensors 
    # each list is 1 img in btch
    def get_pos_embedding(self, pl, ogs) :
        # print('in get_pos_embedding', pl, ogs)
        # print('\n')
        pos = []
        lbls = []
        # print('lines in point list', len(pl))
        # print('shapes of chart', len(ogs))
        for l_ix, lines in enumerate(pl):
            ids = []
            osz = ogs[l_ix]
            for pt_ix, pt in enumerate(lines) : 
                # print('pt in lines', pt)
                pt_x, pt_y = pt[0], pt[1]
                if ogs is not None : 
                    pt_x/=osz[0]
                    pt_x*=128
                    pt_y/=osz[1]
                    pt_y*=128
                if pt_x >=128 :
                    print('::: ERROR ::: x', pt_x, osz) 
                    pt_x = 127
                if pt_y>=128 :
                    print('::: ERROR ::: y', pt_y, osz) 
                    pt_y = 127
                # print('pt_x, pt_y', pt_x, pt_y)
                coord_ix =  self.coord_range[(int(pt_x), int(pt_y))] ## max 128*128 = 16384
                # print('coord_ix',coord_ix)
                # w, h = sz[pt_ix]    
                # ofx, ofy = (w/128) - math.floor((w/128)) , (h/128)-math.floor((h/128))
                # of_ix = self.get_of_ix((ofx, ofy))
                # ids.append((coord_ix*10)+of_ix)
                ids.append(coord_ix)
            lbl = torch.tensor(ids).long().cuda()
            embeds = self.e_lookup(lbl)
            # # print('embeds', embeds.shape)
            # if embeds.shape[0] > self.max_points :
                # embeds = embeds[:self.max_points, :]
            # if embeds.shape[0] < self.max_points :
            #     # print('add padd', self.max_points - embeds.shape[0], embeds.shape[1])
            #     pad_points = torch.zeros(self.max_points - embeds.shape[0], embeds.shape[1]).cuda()
            #     embeds = torch.cat((embeds, pad_points))
            #     # print('embeds', embeds.shape)
            lbls.append(lbl)
            pos.append(embeds)
        # print('pos', len(pos))
        
        return pos, lbls
    def get_of_ix(self, ofx, ofy):
        a = 0
        for k in self.ofx_range :
            if ofx > self.ofx_range[k][0] and ofx<= self.ofx_range[k][1] :
                a+=k
        for k in self.ofy_range :
            if ofy > self.ofy_range[k][0] and ofy<= self.ofy_range[k][1] :
                a+=k
        return a 
  

    def encode_lines(self, sequence, sr_mask, pad_mask) :
        return self.encoder(sequence , mask=sr_mask, src_key_padding_mask=pad_mask)
    
    def decode_lines(self, deq, memory, src_mask, pad_mask) :
        deq = self.decoder(deq, memory, tgt_mask=src_mask, tgt_key_padding_mask=pad_mask)
        # print('decoded', deq.shape)
        return deq
    
    def decode_lines_pointwise(self, deq, memory, src_mask, pad_mask) :
        # deq = self.decoder(deq, memory, tgt_mask=src_mask, tgt_key_padding_mask=pad_mask)
        # print('decode_lines_pointwise', deq.shape, memory.shape)
        # print(pad_mask)
        # run = True 
        # start = self.e_lookup(self.start_token).unsqueeze(0).unsqueeze(0).repeat(deq.shape[0], 1, 1)
        decoded = []
        # print('decoded', decoded)
        for pt_idx, seq in enumerate(range(deq.shape[1])) : 
            # line = [self.e_lookup(self.start_token).unsqueeze(0).unsqueeze(0)]
            line = [self.start_token]
            for steps in range(1, self.max_points):
                trg_tensor = torch.LongTensor(line).unsqueeze(1).cuda() ## ids 
                print('\n tr', trg_tensor.shape)
                trg_tensor = self.e_lookup(trg_tensor) ## emb
                print('tr', trg_tensor.shape)
                sentence_tensor = memory[pt_idx, :, :]
                print('st', sentence_tensor.shape)
                sentence_tensor = sentence_tensor.unsqueeze(0)
                print('st', sentence_tensor.shape)
                deq = self.decoder(trg_tensor, sentence_tensor)
                print('op deq', deq.shape)
                deq = self.project(deq)
                print('projected deq', deq.shape)

                best_guess = deq.argmax(2)[-1, :].item()
                print(best_guess)
                line.append(best_guess)
                if best_guess == self.stop_token :
                    break
            print('line', len(line), line)
            decoded.append(line[1:])
        
        print('decoded', len(decoded))
        lines_coord = []
        for d in decoded :
            lines_coord.append([])
            for key in d : 
                lines_coord[-1].append(self.coord_range_rev[key])
       
        return lines_coord



def ptf(**kwd):
    model = PtFormer(config=kwd['config'])
    return model

# point_list = [[(1,2), (3,4)], [(5,6), (7,8)]]

# total_params = 0
# for params in p.parameters():
#     num_params = 1
#     for x in params.size():
#         num_params *= x
#     total_params += num_params

 