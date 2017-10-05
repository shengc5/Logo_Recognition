# LogoTraining.py
# TianYang Jin, Sheng, Chen
# CSE 415 Project
#

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


from scipy import ndimage
from scipy import misc

from skimage import data, io, transform
from skimage.viewer import ImageViewer
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import rgb2grey, label2rgb
from skimage.util import pad

image = io.imread('sampleBenz.jpg')
#image = io.imread('LexusLogo.png')
greyImage = rgb2grey(image)

#greyImage = rgb2grey(image)
#print(greyImage)
#imgout = greyImage == 1
#io.imshow(imgout)
threshold = threshold_otsu(greyImage)
#imgout = greyImage > threshold
imgout = closing(greyImage > threshold, square(1))


def crop(a):
    minr = 0
    for r in range(a.shape[0]):
        if all(a[r, :] == 1):
            minr += 1
        else:
            break
    maxr = a.shape[0]-1
    for r in range(a.shape[0]-1, -1, -1):
        if all(a[r, :] == 1):
            maxr -= 1
        else:
            break
    minc = 0
    for c in range(a.shape[1]):
        if all(a[:, c] == 1):
            minc += 1
        else:
            break
    maxc = a.shape[1]-1
    for c in range(a.shape[1]-1, -1, -1):
        if all(a[:, c] == 1):
            maxc -= 1
        else:
            break
    return a[minr:maxr, minc:maxc]


def pad_to_square(a, pad_value=1):
    h = a.shape[0]
    w = a.shape[1]
    padded = pad_value * np.ones(2 * [w], dtype=a.dtype)
    padded[(w-h)/2:(w-h)/2+h, 0:w] = a
    return padded


#
print(max(imgout.shape))
imgout = crop(imgout)
imgout = transform.resize(imgout, (max(imgout.shape), max(imgout.shape)))
#imgout = pad_to_square(crop(imgout))

io.imshow(imgout)
io.show()