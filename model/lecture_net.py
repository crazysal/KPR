sftp://hal.cedar.buffalo.edu/home/sahmed9/reps/KeyPointRelations/data/lectureMath/IEEE_access_data_release


import PIL 
from PIL import Image
from scipy.ndimage import gaussian_filter
img =  Image.open('/home/sahmed9/reps/KeyPointRelations/data/lectureMath/IEEE_access_data_release/LectureMath_00000_000_001/binary/6442.png')

def single_norm_n_flip(img) :
    if len(img.shape) > 2 :
        img = img[:, :, 0]
    img = img/max(img)
    img = 1 - img 
    return img 

def get_masks(img):
    img_g = gaussian_filter(img)

f, axarr = plt.subplots(2,2) 
axarr[0][0].imshow(im_n)
im_ = gaussian_filter(im_n, 1.5)
im_.sum()
np.unique(im_[im_ > 0]).shape
axarr[0][1].imshow(im_)
a = torch.sigmoid(torch.tensor(im_n)).numpy()
axarr[1][1].imshow(a)
nz_val = sorted(np.unique(im_[im_ > 0]))
l8 = int(len(nz_val) * 0.85)
im_[im_ <= nz_val[l8] ]  = 0
im_.sum() 
np.unique(im_[im_ > 0]).shape
axarr[1][0].imshow(im_)
plt.show()

np.unique(im_n[im_n> 0]).shape
np.unique(im_[im_ > 0]).shape


f, axarr = plt.subplots(2,2) 
axarr[0][0].imshow(im_n)
axarr[0][1].imshow(gaussian_filter(im_n, 3))
axarr[1][0].imshow(gaussian_filter(im_n, 4))
axarr[1][1].imshow(gaussian_filter(im_n, 5))
plt.show()