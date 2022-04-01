
import numpy as np
import torchvision.transforms as T
import torch 
'''a = torch.rand(20, 20)



b = np.array([
    [10, 12,  8,  7],
    [ 4, 11,  5,  9],
    [18, 13,  7,  7],
    [ 3, 15,  2,  2]
])

from skimage import data, exposure
image = data.astronaut()
an = image[:, :, 0]
an = np.array([
    [5,4,3,2,1],
    [1,2,3,4,5],
    [1,2,5,4,5],
    [4,6,5,3,5],
    [7,6,2,4,5],
])

a = torch.tensor(an)
a = a.unsqueeze(0).unsqueeze(0)
'''
def max_pooling(img: np.array, pool_size: int, stride: int) -> np.array:
    # To store individual pools
    pools = []    
    # Iterate over all row blocks (single block has `stride` rows)
    for i in np.arange(img.shape[0], step=stride):
        # Iterate over all column blocks (single block has `stride` columns)
        for j in np.arange(img.shape[0], step=stride):            
            # Extract the current pool
            mat = img[i:i+pool_size, j:j+pool_size]            
            # Make sure it's rectangular - has the shape identical to the pool size
            if mat.shape == (pool_size, pool_size):
                # Append to the list of pools
                pools.append(mat)                
    # all pools as a Numpy array
    pools =  np.array(pools)
    # Total number of pools
    num_pools = pools.shape[0]
    # Shape of the matrix after pooling - Square root of the number of pools
    # Cast it to int, as Numpy will return it as float
    # For example -> np.sqrt(16) = 4.0 -> int(4.0) = 4
    tgt_shape = (int(np.sqrt(num_pools)), int(np.sqrt(num_pools)))
    # To store the max values
    pooled = []    
    # Iterate over all pools
    for pool in pools:
        # Append the max value only
        pooled.append(np.max(pool))        
    # Reshape to target shape
    return np.array(pooled).reshape(tgt_shape)

## input = NCHW tensor
def top_pool(img) :
    h = img.shape[2]-1
    op = torch.zeros_like(img)
    pre = img.select(2, h)    
    op[:, : , h, : ]  = pre
    for i in range(1, h+1):
        # print('i, h, h-i', i, h-i)
        current = img.select(2, h-i)
        # print('cur', current)
        # print('pre', pre)
        op[:, : , h-i, : ]  = torch.maximum(pre, current)
        pre = op.clone().select(2, h-i)
        # print('pre', pre)
        # print('\n')
    return op      

## input = NCHW tensor
def bottom_pool(img) :
    h = img.shape[2]-1
    op = torch.zeros_like(img)
    pre = img.select(2, 0)    
    op[:, : , 0, : ]  = pre
    for i in range(1, h+1):
        # print('i, h, h-i', i, h-i)
        current = img.select(2, i)
        # print('cur', current)
        # print('pre', pre)
        op[:, : , i, : ]  = torch.maximum(pre, current)
        pre = op.clone().select(2, i)
        # print('pre', pre)
        # print('\n')
    return op      


## input = NCHW tensor
def right_pool(img) :
    w = img.shape[3]-1
    op = torch.zeros_like(img)
    pre = img.select(3, 0)    
    op[:, : , :, 0 ]  = pre
    for i in range(1, w+1):
        # print('i, h, h-i', i, h-i)
        current = img.select(3, i)
        # print('cur', current)
        # print('pre', pre)
        op[:, : , :, i ]  = torch.maximum(pre, current)
        pre = op.clone().select(3, i)
        # print('pre', pre)
        # print('\n')
    return op      


## input = NCHW tensor
def left_pool(img) :
    w = img.shape[3]-1
    op = torch.zeros_like(img)
    pre = img.select(3, w)    
    op[:, : , :, w ]  = pre
    for i in range(1, w+1):
        # print('i, h, h-i', i, h-i)
        current = img.select(3, w-i)
        # print('cur', current)
        # print('pre', pre)
        op[:, : , :, w-i ]  = torch.maximum(pre, current)
        pre = op.clone().select(3, w-i)
        # print('pre', pre)
        # print('\n')
    return op      


# for given arr return elementwise 
# maximum in ascending index of arr
def convert_to_max(arr) :
    p = arr[0]
    a_ = [p]
    for i in range(1, len(arr)) :
        c = arr[i]
        a_.append(max(p, c))
        p = max(a_)
    return np.array(a_)



def diagonals_to_marix(d_):
    sz = len(d_)
    row = int((sz -1) /2)
    m = np.zeros((row+1, row+1))
    # print(m, m.shape)
    for ix, arr in enumerate(d_):
        if ix <= row : 
            r = row-ix
            c = 0 
        else: 
            r = 0
            c = ix-row
        for a in arr :
            # print(a, r, c)
            m[r][c] = a
            r+=1
            c+=1
    return m




#top-left to botom-right
# img is HW array
def pos_diagonal(img) :
    w = img.shape[1] - 1
    d = [] 
    d_r = ((-1*w), w+1)
    # print('got dr', d_r)
    for i in range(d_r[0], d_r[1]):
        d_ = img.diagonal(i)
        d.append(convert_to_max(d_))
    # print('got d', d)
    return diagonals_to_marix(d)




# botom-right to top-left
# img is HW array
def neg_diagonal(img) :
    w = img.shape[1] - 1
    d = [] 
    d_r = ((-1*w), w+1)
    # print('got dr', d_r)
    for i in range(d_r[0], d_r[1]):
        d_ = img.diagonal(i)
        d.append(reversed(convert_to_max(list(reversed(d_)))))
    # print('got d', d)
    return diagonals_to_marix(d)



#top-left to botom-right
# img is HW array
def pos_anti_diagonal(img) :
    img = np.rot90(img)
    w = img.shape[1] - 1
    d = [] 
    d_r = ((-1*w), w+1)
    # print('got dr', d_r)
    for i in range(d_r[0], d_r[1]):
        d_ = img.diagonal(i)
        d.append(convert_to_max(d_))
    # print('got d', d)
    return np.rot90(np.rot90(np.rot90(diagonals_to_marix(d))))



# botom-right to top-left
# img is HW array
def neg_anti_diagonal(img) :
    img = np.rot90(img)
    w = img.shape[1] - 1
    d = [] 
    d_r = ((-1*w), w+1)
    # print('got dr', d_r)
    for i in range(d_r[0], d_r[1]):
        d_ = img.diagonal(i)
        d.append(reversed(convert_to_max(list(reversed(d_)))))
    # print('got d', d)
    return np.rot90(np.rot90(np.rot90(diagonals_to_marix(d))))
    


# for i in range(-3, 4):
#     print(i, b_r.diagonal(i) , '\n')

# for i in range(-3, 4):
#     print(i, b.diagonal(i) , '\n')


# for given arr return elementwise 
# maximum in ascending index of tsr
# tsr : NCx ; x = length of diagonal
def convert_to_max_tensor(tsr) :
    sz = tsr.shape[-1]
    op = torch.zeros_like(tsr)
    op[:, :, 0] = tsr[:, :, 0]
    pre = tsr.select(-1, 0)
    for i in range(1, sz) :
        curr = tsr.select(-1, i)
        op[:, :, i] = torch.maximum(pre, curr)
        pre = op.clone().select(-1, i)
    return op

def diagonals_to_matrix_tensor(d_, m):
    sz = len(d_)
    row = int((sz -1) /2)
    # print(m, m.shape)
    for ix, tensor in enumerate(d_):
        # print(ix, tensor, tensor.shape)
        if ix <= row : 
            r = row-ix
            c = 0 
        else: 
            r = 0
            c = ix-row
        for last_ix in range(tensor.shape[-1]) :
            m[:, :, r, c] = tensor[:, :, last_ix]
            r+=1
            c+=1
    return m


def pos_diagonal_tensor(img) :
    # print('got img', img, img.shape)
    w = img.shape[-1] - 1
    d = [] 
    d_r = ((-1*w), w+1)
    # print('got dr', d_r)
    m = torch.zeros_like(img)
    for i in range(d_r[0], d_r[1]):
        d_ = img.diagonal(i, -2, -1)
        if d_.shape[-1] > 1 :
            d_ = convert_to_max_tensor(d_) 
        d.append(d_)
        # print('got d', d)
    return diagonals_to_matrix_tensor(d, m)
 

def neg_diagonal_tensor(img) :
    # print('got img', img, img.shape)
    w = img.shape[-1] - 1
    d = [] 
    d_r = ((-1*w), w+1)
    # print('got dr', d_r)
    m = torch.zeros_like(img)
    for i in range(d_r[0], d_r[1]):
        d_ = img.diagonal(i, -2, -1)
        if d_.shape[-1] > 1 :
            d__ = []
            for k in d_ :
                d__.append(k.fliplr())
            d_ = torch.stack(d__)
            d_ = convert_to_max_tensor(d_) 
            d__ = []
            for k in d_ :
                d__.append(k.fliplr())
            d_ = torch.stack(d__)
        d.append(d_)
        # print('got d', d)
    return diagonals_to_matrix_tensor(d, m)


def pos_anti_diagonal_tensor(img) :
    # print('got img', img, img.shape)
    img = T.functional.rotate(img, 90)
    w = img.shape[-1] - 1
    d = [] 
    d_r = ((-1*w), w+1)
    # print('got dr', d_r)
    m = torch.zeros_like(img)
    for i in range(d_r[0], d_r[1]):
        d_ = img.diagonal(i, -2, -1)
        if d_.shape[-1] > 1 :
            d_ = convert_to_max_tensor(d_) 
        d.append(d_)
        # print('got d', d)
    return T.functional.rotate(diagonals_to_matrix_tensor(d, m), -90)

 
def neg_anti_diagonal_tensor(img) :
    # print('got img', img, img.shape)
    img = T.functional.rotate(img, 90)
    w = img.shape[-1] - 1
    d = [] 
    d_r = ((-1*w), w+1)
    # print('got dr', d_r)
    m = torch.zeros_like(img)
    for i in range(d_r[0], d_r[1]):
        d_ = img.diagonal(i, -2, -1)
        if d_.shape[-1] > 1 :
            d__ = []
            for k in d_ :
                d__.append(k.fliplr())
            d_ = torch.stack(d__)
            d_ = convert_to_max_tensor(d_) 
            d__ = []
            for k in d_ :
                d__.append(k.fliplr())
            d_ = torch.stack(d__)
        d.append(d_)
        # print('got d', d)
    return T.functional.rotate(diagonals_to_matrix_tensor(d, m), -90)


'''

image = data.astronaut()
i1 = max_pooling(image[: ,:, 0])
i2 = max_pooling(image[: ,:, 1])
i3 = max_pooling(image[: ,:, 2])
an = image[: ,:, 0]

im = cv2.imread('download.png')#in CRAFT dir
an = im[10:223, 50:263, 0]
an = an /255
an = 1 -an 

a = torch.tensor(an)
a = a.unsqueeze(0).unsqueeze(0)


fig, axs = plt.subplots(6, 6)

axs[0, 0].imshow(an/a.max().item())
axs[0, 0].set_title('Img')
m = max_pooling(an, 2, 1)/a.max().item()
axs[0, 1].imshow(m)
axs[0, 1].set_title('Max pool')

t =top_pool(a)[0, 0, :, :].numpy()/a.max().item()
axs[0, 2].imshow(t)
axs[0, 2].set_title('t')

b =bottom_pool(a)[0, 0, :, :].numpy()/a.max().item()
axs[0, 3].imshow(b)
axs[0, 3].set_title('b')

l =left_pool(a)[0, 0, :, :].numpy()/a.max().item()
axs[0, 4].imshow(l)
axs[0, 4].set_title('l')

r =right_pool(a)[0, 0, :, :].numpy()/a.max().item()
axs[0, 5].imshow(r)
axs[0, 5].set_title('r')


pd = pos_diagonal(an)/a.max().item()
axs[3, 0].imshow(pd)
axs[3, 0].set_title('PD')

nd = neg_diagonal(an)/a.max().item()
axs[3, 1].imshow(nd)
axs[3, 1].set_title('ND')

pad = pos_anti_diagonal(an)/a.max().item()
axs[3, 2].imshow(pad)
axs[3, 2].set_title('PAD')

nad = neg_anti_diagonal(an)/a.max().item()
axs[3, 3].imshow(nad)
axs[3, 3].set_title('NAD')

pdt = (pd+t)/2
axs[3, 4].imshow(pdt)
axs[3, 4].set_title('pdt')

nadt = (nad+t)/2
axs[3, 5].imshow(nadt)
axs[3, 5].set_title('nadt')




tb =(t+b)/2
axs[1, 0].imshow(tb)
axs[1, 0].set_title('tb')
lr =(l+r)/2
axs[1, 1].imshow(lr)
axs[1, 1].set_title('lr')

tl =(t+l)/2
axs[1, 2].imshow(tl)
axs[1, 2].set_title('tl')
tr =(t+r)/2
axs[1, 3].imshow(tr)
axs[1, 3].set_title('tr')

bl =(b+l)/2
axs[1, 4].imshow(bl)
axs[1, 4].set_title('bl')

br =(b+r)/2
axs[1, 5].imshow(br)
axs[1, 5].set_title('bl')

tbl =(t+b+l)/3
axs[2, 0].imshow(tbl)
axs[2, 0].set_title('tbl')

tbr =(t+b+r)/3
axs[2, 1].imshow(tbr)
axs[2, 1].set_title('tbr')

tlr =(t+r+l)/3
axs[2, 2].imshow(tlr)
axs[2, 2].set_title('tlr')

blr =(l+b+r)/3
axs[2, 3].imshow(blr)
axs[2, 3].set_title('blr')

tblr =(t+l+b+r)/4
axs[2, 4].imshow(tblr)
axs[2, 4].set_title('tblr')

## ----- 
pdnd =(pd+nd)/2
axs[4, 0].imshow(pdnd)
axs[4, 0].set_title('pdnd')
padnad =(pad+nad)/2
axs[4, 1].imshow(padnad)
axs[4, 1].set_title('padnad')

pdnad =(pd+nad)/2
axs[4, 2].imshow(pdnad)
axs[4, 2].set_title('pdnad')
padnd =(pad+nd)/2
axs[4, 3].imshow(padnad)
axs[4, 3].set_title('padnad')

pdpad =(pd+pad)/2
axs[4, 4].imshow(pdpad)
axs[4, 4].set_title('pdpad')

ndnad =(nd+nad)/2
axs[4, 5].imshow(ndnad)
axs[4, 5].set_title('ndnad')

pdpadnad =(pd+pad+nad)/3
axs[5, 0].imshow(pdpadnad)
axs[5, 0].set_title('pdpadnad')

ndpadnad =(nd+pad+nad)/3
axs[5, 1].imshow(ndpadnad)
axs[5, 1].set_title('ndpadnad')

pdndnad =(pd+nd+nad)/3
axs[5, 2].imshow(pdndnad)
axs[5, 2].set_title('pdndnad')

pdndpad =(pd+nd+pad)/3
axs[5, 3].imshow(pdndnad)
axs[5, 3].set_title('pdndnad')

pdndpadnad =(pd+nd+pad+nad)/4
axs[5, 4].imshow(pdndpadnad)
axs[5, 4].set_title('pdndpadnad')


pdndpadnadtlbr =(pd+nd+pad+nad+t+l+b+r)/8
axs[5, 5].imshow(pdndpadnadtlbr)
axs[5, 5].set_title('pdndpadnadtlbr')



>>> b
array([[10, 12,  8,  7],
       [ 4, 11,  5,  9],
       [18, 13,  7,  7],
       [ 3, 15,  2,  2]])
pd
array([[10., 12.,  8.,  7.],
       [ 4., 11., 12.,  9.],
       [18., 13., 11., 12.],
       [ 3., 18., 13., 11.]])
negd
array([[11., 12.,  9.,  7.],
       [13., 11.,  7.,  9.],
       [18., 13.,  7.,  7.],
       [ 3., 15.,  2.,  2.]])

pos_antd 
array([[10., 12.,  8.,  7.],
       [12., 11.,  7.,  9.],
       [18., 13.,  9.,  7.],
       [13., 15.,  7.,  2.]])

neg_antd
array([[10., 12., 18., 13.],
       [ 4., 18., 13., 15.],
       [18., 13., 15.,  7.],
       [ 3., 15.,  2.,  2.]])
'''

'''

fig, axs = plt.subplots(6, 6)

axs[0, 0].imshow(a.squeeze(0).numpy()/a.max().item())
axs[0, 0].set_title('Img')

# m = max_pooling(a.squeeze(0).numpy(), 2, 1)/a.max().item()
# axs[0, 1].imshow(m)
# axs[0, 1].set_title('Max pool')

t = top_pool(a).squeeze(0).numpy()/a.max().item()
axs[0, 2].imshow(t)
axs[0, 2].set_title('t')

b =bottom_pool(a).squeeze(0).numpy()/a.max().item()
axs[0, 3].imshow(b)
axs[0, 3].set_title('b')

l= left_pool(a).squeeze(0).numpy()/a.max().item()
axs[0, 4].imshow(l)
axs[0, 4].set_title('l')

r =right_pool(a).squeeze(0).numpy()/a.max().item()
axs[0, 5].imshow(r)
axs[0, 5].set_title('r')


pd = pos_diagonal_tensor(a).squeeze(0)/a.max().item()
axs[3, 0].imshow(pd)
axs[3, 0].set_title('PD')

nd = neg_diagonal_tensor(a).squeeze(0)/a.max().item()
axs[3, 1].imshow(nd)
axs[3, 1].set_title('ND')

pad = pos_anti_diagonal_tensor(a).squeeze(0)/a.max().item()
axs[3, 2].imshow(pad)
axs[3, 2].set_title('PAD')

nad = neg_anti_diagonal_tensor(a).squeeze(0)/a.max().item()
axs[3, 3].imshow(nad)
axs[3, 3].set_title('NAD')


'''
'''
tensor([[[[ 22, 210, 216],
          [207, 168, 158],
          [ 36,  65, 142]],

         [[109, 114, 136],
          [229, 214,  22],
          [204, 103, 192]],

         [[ 31, 198, 239],
          [129,  98, 225],
          [ 59, 135, 147]]],


        [[[ 66, 235,  42],
          [103,  89, 148],
          [ 59, 170, 244]],

         [[ 29,  41, 218],
          [216, 152, 111],
          [120, 178,  70]],

         [[230, 228, 230],
          [156, 224, 186],
          [ 32, 105,  40]]],


        [[[116,  84,  70],
          [206, 237,  84],
          [196,   9,  57]],

         [[ 99, 180, 239],
          [ 35, 103, 196],
          [137, 141, 234]],

         [[ 26, 178, 181],
          [208, 196, 192],
          [ 78, 183,  27]]]], dtype=torch.int32)

>>> top_pool(s)
tensor([[[[207, 210, 216],
          [207, 168, 158],
          [ 36,  65, 142]],

         [[229, 214, 192],
          [229, 214, 192],
          [204, 103, 192]],

         [[129, 198, 239],
          [129, 135, 225],
          [ 59, 135, 147]]],


        [[[103, 235, 244],
          [103, 170, 244],
          [ 59, 170, 244]],

         [[216, 178, 218],
          [216, 178, 111],
          [120, 178,  70]],

         [[230, 228, 230],
          [156, 224, 186],
          [ 32, 105,  40]]],


        [[[206, 237,  84],
          [206, 237,  84],
          [196,   9,  57]],

         [[137, 180, 239],
          [137, 141, 234],
          [137, 141, 234]],

         [[208, 196, 192],
          [208, 196, 192],
          [ 78, 183,  27]]]], dtype=torch.int32)

>>> bottom_pool(s)
tensor([[[[ 22, 210, 216],
          [207, 210, 216],
          [207, 210, 216]],

         [[109, 114, 136],
          [229, 214, 136],
          [229, 214, 192]],

         [[ 31, 198, 239],
          [129, 198, 239],
          [129, 198, 239]]],


        [[[ 66, 235,  42],
          [103, 235, 148],
          [103, 235, 244]],

         [[ 29,  41, 218],
          [216, 152, 218],
          [216, 178, 218]],

         [[230, 228, 230],
          [230, 228, 230],
          [230, 228, 230]]],


        [[[116,  84,  70],
          [206, 237,  84],
          [206, 237,  84]],

         [[ 99, 180, 239],
          [ 99, 180, 239],
          [137, 180, 239]],

         [[ 26, 178, 181],
          [208, 196, 192],
          [208, 196, 192]]]], dtype=torch.int32)
>>> left_pool(s)
tensor([[[[216, 216, 216],
          [207, 168, 158],
          [142, 142, 142]],

         [[136, 136, 136],
          [229, 214,  22],
          [204, 192, 192]],

         [[239, 239, 239],
          [225, 225, 225],
          [147, 147, 147]]],


        [[[235, 235,  42],
          [148, 148, 148],
          [244, 244, 244]],

         [[218, 218, 218],
          [216, 152, 111],
          [178, 178,  70]],

         [[230, 230, 230],
          [224, 224, 186],
          [105, 105,  40]]],


        [[[116,  84,  70],
          [237, 237,  84],
          [196,  57,  57]],

         [[239, 239, 239],
          [196, 196, 196],
          [234, 234, 234]],

         [[181, 181, 181],
          [208, 196, 192],
          [183, 183,  27]]]], dtype=torch.int32)
>>> right_pool(s)
tensor([[[[ 22, 210, 216],
          [207, 207, 207],
          [ 36,  65, 142]],

         [[109, 114, 136],
          [229, 229, 229],
          [204, 204, 204]],

         [[ 31, 198, 239],
          [129, 129, 225],
          [ 59, 135, 147]]],


        [[[ 66, 235, 235],
          [103, 103, 148],
          [ 59, 170, 244]],

         [[ 29,  41, 218],
          [216, 216, 216],
          [120, 178, 178]],

         [[230, 230, 230],
          [156, 224, 224],
          [ 32, 105, 105]]],


        [[[116, 116, 116],
          [206, 237, 237],
          [196, 196, 196]],

         [[ 99, 180, 239],
          [ 35, 103, 196],
          [137, 141, 234]],

         [[ 26, 178, 181],
          [208, 208, 208],
          [ 78, 183, 183]]]], dtype=torch.int32)

>>> pos_diagonal_tensor(s)
tensor([[[[ 22, 210, 216],
          [207, 168, 210],
          [ 36, 207, 168]],

         [[109, 114, 136],
          [229, 214, 114],
          [204, 229, 214]],

         [[ 31, 198, 239],
          [129,  98, 225],
          [ 59, 135, 147]]],


        [[[ 66, 235,  42],
          [103,  89, 235],
          [ 59, 170, 244]],

         [[ 29,  41, 218],
          [216, 152, 111],
          [120, 216, 152]],

         [[230, 228, 230],
          [156, 230, 228],
          [ 32, 156, 230]]],


        [[[116,  84,  70],
          [206, 237,  84],
          [196, 206, 237]],

         [[ 99, 180, 239],
          [ 35, 103, 196],
          [137, 141, 234]],

         [[ 26, 178, 181],
          [208, 196, 192],
          [ 78, 208, 196]]]], dtype=torch.int32)

>>> neg_diagonal_tensor(s)
tensor([[[[168, 210, 216],
          [207, 168, 158],
          [ 36,  65, 142]],

         [[214, 114, 136],
          [229, 214,  22],
          [204, 103, 192]],

         [[147, 225, 239],
          [135, 147, 225],
          [ 59, 135, 147]]],


        [[[244, 235,  42],
          [170, 244, 148],
          [ 59, 170, 244]],

         [[152, 111, 218],
          [216, 152, 111],
          [120, 178,  70]],

         [[230, 228, 230],
          [156, 224, 186],
          [ 32, 105,  40]]],


        [[[237,  84,  70],
          [206, 237,  84],
          [196,   9,  57]],

         [[234, 196, 239],
          [141, 234, 196],
          [137, 141, 234]],

         [[196, 192, 181],
          [208, 196, 192],
          [ 78, 183,  27]]]], dtype=torch.int32)

>>> pos_anti_diagonal_tensor(s)
tensor([[[[ 22, 210, 216],
          [210, 216, 158],
          [216, 158, 142]],

         [[109, 114, 136],
          [229, 214,  22],
          [214, 103, 192]],

         [[ 31, 198, 239],
          [198, 239, 225],
          [239, 225, 147]]],


        [[[ 66, 235,  42],
          [235,  89, 148],
          [ 89, 170, 244]],

         [[ 29,  41, 218],
          [216, 218, 111],
          [218, 178,  70]],

         [[230, 228, 230],
          [228, 230, 186],
          [230, 186,  40]]],


        [[[116,  84,  70],
          [206, 237,  84],
          [237,  84,  57]],

         [[ 99, 180, 239],
          [180, 239, 196],
          [239, 196, 234]],

         [[ 26, 178, 181],
          [208, 196, 192],
          [196, 192,  27]]]], dtype=torch.int32)
>>> neg_anti_diagonal_tensor(s)
tensor([[[[ 22, 210, 216],
          [207, 168, 158],
          [ 36,  65, 142]],

         [[109, 229, 214],
          [229, 214, 103],
          [204, 103, 192]],

         [[ 31, 198, 239],
          [129,  98, 225],
          [ 59, 135, 147]]],


        [[[ 66, 235,  89],
          [103,  89, 170],
          [ 59, 170, 244]],

         [[ 29, 216, 218],
          [216, 152, 178],
          [120, 178,  70]],

         [[230, 228, 230],
          [156, 224, 186],
          [ 32, 105,  40]]],


        [[[116, 206, 237],
          [206, 237,  84],
          [196,   9,  57]],

         [[ 99, 180, 239],
          [ 35, 137, 196],
          [137, 141, 234]],

         [[ 26, 208, 196],
          [208, 196, 192],
          [ 78, 183,  27]]]], dtype=torch.int32)


>>> s.shape
torch.Size([5, 256, 224, 224])

t1 = time.time(); s_ = top_pool(s); t2 = time.time(); t2-t1
t1 = time.time(); s_ = bottom_pool(s); t2 = time.time(); t2-t1
t1 = time.time(); s_ = left_pool(s); t2 = time.time(); t2-t1
t1 = time.time(); s_ = right_pool(s); t2 = time.time(); t2-t1

t1 = time.time(); s_ = neg_diagonal_tensor(s); t2 = time.time(); t2-t1
t1 = time.time(); s_ = pos_diagonal_tensor(s); t2 = time.time(); t2-t1
t1 = time.time(); s_ = neg_anti_diagonal_tensor(s); t2 = time.time(); t2-t1
t1 = time.time(); s_ = pos_anti_diagonal_tensor(s); t2 = time.time(); t2-t1
'''