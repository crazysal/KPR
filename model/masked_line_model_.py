
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import copy 

from model.hg_net import HourglassNet, Bottleneck
class Mlm(nn.Module):
    def __init__(self, **kwd):
        super(Mlm, self).__init__()
        config = kwd['config']
        self.max_points = 40
        self.max_lines = 8
        self.d_model = 256
        self.ntoken = 128*128+2
        self.start_token = torch.tensor(16384).cuda()
        self.stop_token = torch.tensor(16385).cuda()
        chnl_ = config.heatmap_chan = 256
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
        # print('self.coord_range', self.coord_range)
        # print(self.e_lookup)
        self.coord_range_rev = {}
        key = 0
        for x_ in range(128) :
            for y_ in range(128) :
                self.coord_range_rev.update({key: (x_, y_)})
                key+=1
        self.coord_range_rev.update({16384: 'start'})
        self.coord_range_rev.update({16385: 'stop'})
        # print('self.coord_range_rev', self.coord_range_rev)
        # exit()
        # self.ofx_range = {0 :(0, 0), 1 :(0, 0.2), 2 :(0.2, 0.4), 3 :(0.4, 0.6), 4 :(0.6, 0.8), 5 :(0.8, 1) }
        # self.ofy_range = {0 :(0, 0), 1 :(0, 0.2), 2 :(0.2, 0.4), 3 :(0.4, 0.6), 4 :(0.6, 1)}
    def forward(self, x, edge_list, point_list_, point_norm_list, split, _):
        decoder_loss = _
        x, x_ = self.hg(x)
        if decoder_loss :
            seq, src_mask, pad_mask = self.get_encoder_sequence(x, x_, point_list_, point_norm_list,  )
            memory = self.encode_lines(seq, src_mask, pad_mask)
            deq, tgt_mask, tpad_mask, labels = self.get_decoder_sequence(point_list_)
            if split !='test' : 
                output = self.decode_lines(deq[:-1, :, :], memory, tgt_mask[:-1, :-1], tpad_mask[:, :-1])
                op_deq = self.project(output)
                op_coord = self.project_coord(output)
                return {'hm': x, 'deq' :op_deq, 'cor': op_coord , 'pnl': (point_norm_list, point_list_[1]), 'type' : 'transformer', 'decoder_loss':decoder_loss, 'dec_label':labels}
            else : 
                output=  self.decode_lines_pointwise(deq, memory, tgt_mask, tpad_mask)
                exit()
                return output

        else :
            return {'hm': x, 'type' : 'transformer', 'decoder_loss':decoder_loss}

    def get_decoder_sequence(self, point_list_, og_size=None):
        point_list, pt_idx = point_list_ 
        pos_embed, labla = self.get_pos_embedding(point_list, og_size) ## N x model_dim  where N = (total lines in batch)
        feats = []
        masks = []
        labels = [] 
        for p_ix, pos_ in enumerate(pos_embed) :
            pl = point_list[p_ix]
            point_count = pt_idx[p_ix]
            pos_  = pos_.split(point_count)
            labla_  = labla[p_ix].split(point_count)
            nop   = []
            mask  = []
            label = [] 
            for o_idx, o in enumerate(pos_) : 
                s = self.start_token.unsqueeze(0)
                e = self.stop_token.unsqueeze(0)
                if o.shape[0] < self.max_points :  ## add padding for lines and mask 
                    pad = torch.zeros(self.max_points - o.shape[0], o.shape[1]).cuda()
                    nop.append(torch.cat((self.e_lookup(s), o, self.e_lookup(e), pad)))
                    pad_id = [16386]*(self.max_points - o.shape[0])
                    pad_id = torch.tensor(pad_id).long().cuda()
                    l_ = torch.cat((labla_[o_idx],e, pad_id))
                    label.append(l_) ## 16386 be pad ID 
                    m = torch.zeros(self.max_points+2).cuda()
                    m[o.shape[0]-(self.max_points):]+=1
                    mask.append(m.bool())
                elif o.shape[0] >= self.max_points :
                    nop.append(torch.cat((self.e_lookup(s),o[:self.max_points, :],self.e_lookup(e))))
                    m = torch.zeros(self.max_points+2).bool().cuda()
                    mask.append(m)
                    l_ = torch.cat((labla_[o_idx][:self.max_points],e))
                    label.append(l_) 
            nop = torch.stack(nop)
            mask = torch.stack(mask)
            label = torch.stack(label)
            feats.append(nop)
            masks.append(mask)
            labels.append(label)
         
        feats = torch.cat(feats, dim=0).permute(1,0,2)
        pad_masks = torch.cat(masks, dim=0)
        tgt_masks = torch.zeros((feats.shape[0], feats.shape[0])).bool().cuda()
        labels = torch.cat(labels)
        return feats , tgt_masks, pad_masks, labels

    def get_encoder_sequence(self, x, x_, point_list_, point_raw_list, og_size=None) :
    
        ## 3 stacks, number output class == 5 
        ## x_ ->> list of outputss channel 256 -> this will give feature embed
        # hm1, hm2, hm3 = x # list of outputss channel 5  -> this will intensity 
        # intensity = torch.sigmoid(heatmaps_pred.squeeze())

        maps = x_[-1]
        point_list, pt_idx = point_list_ 
        print('total point list, pnl', len(point_list),len(point_raw_list))
        tp = 0
        for p_, p__ in zip(point_list, point_raw_list) :
            print('total pts short, raw',len(p_), len(p__))
            tp+=len(p_)
        print('Total Points', tp)
        pos_embed, _ = self.get_pos_embedding(point_list, og_size) ## N x model_dim  where N = (total lines in batch)
        feats = []
        masks = [] 
        for image in range(maps.shape[0]) :
            op = []
            pl = point_list[image]
            pos_ = pos_embed[image]
            point_count = pt_idx[image] 
            for pt_  in pl :
                lines = []
                for pt_ct in point_count :
                    fp = maps[image, :,  pt_[0], pt_[1]]
                op.append(fp)
            op = torch.stack(op)
            op = op.split(point_count)
            pos_ = pos_.split(point_count)
            nop  = []
            mask = []
            for o_idx, o in enumerate(op) : 
                # print('each line feat', o.shape)
                s = self.e_lookup(self.start_token).unsqueeze(0)
                e = self.e_lookup(self.stop_token).unsqueeze(0)
                if o.shape[0] < self.max_points :  ## add padding for lines and mask 
                    pad = torch.zeros(self.max_points - o.shape[0], o.shape[1]).cuda()
                    nop_ = o + pos_[o_idx]
                    nop.append(torch.cat((s, nop_, e, pad)))
                    m = torch.zeros(self.max_points+2).cuda()
                    m[o.shape[0]-(self.max_points):]+=1
                    mask.append(m.bool())
                elif o.shape[0] >= self.max_points :
                    nop_ = o[:self.max_points, :] + pos_[o_idx][:self.max_points, :]
                    nop.append(torch.cat((s, nop_, e)))
                    m = torch.zeros(self.max_points+2).bool().cuda()
                    mask.append(m)

            nop = torch.stack(nop)
            mask = torch.stack(mask)
            feats.append(nop)
            masks.append(mask)
        
        feats = torch.cat(feats, dim=0).permute(1,0,2)
        pad_masks = torch.cat(masks, dim=0)
        src_masks = torch.zeros((feats.shape[0], feats.shape[0])).bool().cuda()
        return feats , src_masks, pad_masks

    ## returns list of tensors 
    # each list is 1 img in btch
    def get_pos_embedding(self, pl, ogs) :
        # print('in get_pos_embedding', pl )
        # print('\n')
        pos = []
        lbls = []
        for l_ix, lines in enumerate(pl):
            ids = []
            # print('points in line', len(lines))
            # sz = ogs[l_ix]
            for pt_ix, pt in enumerate(lines) : 
                # print('pt in lines', pt)
                pt_x, pt_y = pt[0], pt[1]
                # print('pt_x, pt_y', pt_x, pt_y)
                coord_ix =  self.coord_range[(pt_x, pt_y)] ## max 128*128 = 16384
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

 