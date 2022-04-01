import torch
import numpy as np
import torch.nn as nn

# Mangled from : https://github.com/google-research/google-research/blob/559f6170aea8ecf20808f4353e9eff816ebdec24/coltran/models/colorizer.py#L128
'''
# 3 bits per channel, 8 colors per channel, a total of 512 colors.
    self.num_symbols_per_channel = 2**3
    self.num_symbols = self.num_symbols_per_channel**3
    self.gray_symbols, self.num_channels = 256, 1



def image_loss(self, logits, labels):
    """Cross-entropy between the logits and labels."""
    height, width = labels.shape[1:3]
    logits = tf.squeeze(logits, axis=-2)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss = tf.reduce_mean(loss, axis=0)
    loss = base_utils.nats_to_bits(tf.reduce_sum(loss))
    return loss / (height * width)

def convert_bits(x, n_bits_out=8, n_bits_in=8):
    """Quantize / dequantize from n_bits_in to n_bits_out."""
    if n_bits_in == n_bits_out:
        return x
    x = tf.cast(x, dtype=tf.float32)
    x = x / 2**(n_bits_in - n_bits_out)
    x = tf.cast(x, dtype=tf.int32)
    return x

def labels_to_bins(labels, num_symbols_per_channel):
    """Maps each (R, G, B) channel triplet to a unique bin.
    Args:
    labels: 4-D Tensor, shape=(batch_size, H, W, 3).
    num_symbols_per_channel: number of symbols per channel.
    Returns:
    labels: 3-D Tensor, shape=(batch_size, H, W) with 512 possible symbols.
    """
    labels = tf.cast(labels, dtype=tf.float32)
    channel_hash = [num_symbols_per_channel**2, num_symbols_per_channel, 1.0]
    channel_hash = tf.constant(channel_hash)
    labels = labels * channel_hash

    labels = tf.reduce_sum(labels, axis=-1)
    labels = tf.cast(labels, dtype=tf.int32)
    return labels


def loss( targets, logits, train_config, training, aux_output=None):
    """Converts targets to coarse colors and computes log-likelihood."""
    downsample = train_config.get('downsample', False)
    downsample_res = train_config.get('downsample_res', 64)
    if downsample:
      labels = targets['targets_%d' % downsample_res]
    else:
      labels = targets['targets']

    if aux_output is None:
      aux_output = {}

    # quantize labels.
    labels = base_utils.convert_bits(labels, n_bits_in=8, n_bits_out=3)

    # bin each channel triplet.
    labels = base_utils.labels_to_bins(labels, self.num_symbols_per_channel)

    loss = self.image_loss(logits, labels)
    enc_logits = aux_output.get('encoder_logits')
    if enc_logits is None:
      return loss, {}
'''


class Loss(nn.Module):
    def __init__(self, **kwd):
        super(Loss, self).__init__()
        # self.loss = nn.CrossEntropyLoss()
        self.loss = nn.MSELoss(reduction='mean')
    def forward(self, x, y) :
        l = []
        # print('lop loss', y.shape)
        if isinstance (x, list) :
            # for _ in x : 
            #     print('lin loss', _.shape)
            for i in range(len(x)):
                l.append(self.loss(x[i].squeeze(dim=1), y[:, i, :, :]))
        else : 
            # print('lin loss', x.shape)
            for i in range(x.shape[1]):
                l.append(self.loss(x[:, i, :, :], y[:, i, :, :]))
            # print(l)
        return l[0] + l[1] + l[2]




class LossV2(nn.Module):
    def __init__(self, **kwd):
        super(LossV2, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        hash = {}
        for r in range(8):
            for g in range(8):
                for b in range(8): 
                    hash.update({64*r + 8*g +b : (r, g, b)}) 
        self.hash = hash

    @staticmethod
    def convert_bits(x, n_bits_out=8, n_bits_in=8):
        """Quantize / dequantize from n_bits_in to n_bits_out."""
        if n_bits_in == n_bits_out:
            return x
        x = x.float()
        x = x / 2**(n_bits_in - n_bits_out)
        x = x.int()
        return x

    @staticmethod
    def labels_to_bins(labels, num_symbols_per_channel=2**3):
        """Maps each (R, G, B) channel triplet to a unique bin.
        Args:
        labels: 4-D Tensor, shape=(batch_size,3, H, W).
        num_symbols_per_channel: number of symbols per channel.
        Returns:
        labels: 3-D Tensor, shape=(batch_size, H, W) with 512 possible symbols.
        """
        # print('LBL to bin', labels.min(), labels.max(), labels.sum(), labels.shape)
        labels = labels.float()
        channel_hash = torch.tensor([num_symbols_per_channel**2, num_symbols_per_channel, 1.0])    
        # print('channel_hash', channel_hash)
        labels = labels.permute(0, 2, 3, 1) * channel_hash.cuda()
        # print('* hash',labels.min(), labels.max(), labels.sum(), labels.shape)
        labels = labels.sum(dim=-1)
        # print('summed',labels.min(), labels.max(), labels.sum(), labels.shape)
        labels = labels.long()
        # print('int',labels.min(), labels.max(), labels.sum(), labels.shape)
        return labels
    @staticmethod
    def nats_to_bits(nats):
        return nats / np.log(2)


    def image_loss(self, logits, labels):
        """Cross-entropy between the logits and labels."""
        B, height, width = labels.size()
        # print('In image loss, logit, labels =', logits.shape, labels.shape)
        loss = self.loss_fn(logits, labels)
        # print(loss)
        loss = self.nats_to_bits(loss)
        # print(loss)
        loss/= (height * width * B)
        # print(loss)
        return loss

    def forward(self, x, y):
        """
        Args 
        x: logits  4-D Tensor, shape=(batch_size,3, H, W).
        y : labels  4-D Tensor, shape=(batch_size,3, H, W).
        """
        # print('in  loss', x.shape, y.shape)
        # quantize labels.
        if isinstance (x, list) :
            x= x[-1]
        y = self.convert_bits(y, 3, 8)
        # bin each channel triplet.
        y = self.labels_to_bins(y)
        loss = self.image_loss(x, y)
        return loss


