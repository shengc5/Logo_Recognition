# testing.py
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

#image = io.imread('sampleAudi.png')
#image = io.imread('sampleAudi2.jpg')
image = io.imread('sampleChevrolet.jpg')
#image = io.imread('sampleHonda.jpg')
#image = io.imread('sampleBenz.jpg')
#image = io.imread('sampleBMW3.jpg')

greyImage = rgb2grey(image)
greyImage = transform.resize(greyImage, (500, 600))
threshold = threshold_otsu(greyImage)
imageOtsu = closing(greyImage >= threshold, square(3))

# Original Binary
plt.figure()
io.imshow(imageOtsu)

# Edge detection
cpImage = imageOtsu.copy()
clearBorder = clear_border(cpImage)
labeled = label(clearBorder)
borders = np.logical_xor(imageOtsu, cpImage)
labeled[borders] = -1
labeledImage = label2rgb(labeled, image=imageOtsu)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
ax.imshow(labeledImage)

for region in regionprops(labeled):

    if region.area < 100:
        continue

    minr, minc, maxr, maxc = region.bbox
    if (maxr - minr) / (maxc - minc) > 2.5 or (maxc - minc) / (maxr - minr) > 2.5:
        continue

    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='blue', linewidth=1.5)
    ax.add_patch(rect)

plt.show()


# plt.figure()
# io.imshow(labeledImage)
# io.show()