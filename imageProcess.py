# imageProcess.py
# TianYang Jin, Sheng, Chen
# CSE 415 Project
#

import glob
import os
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


def getOutputPath(l, logo, num):
    Path = ''
    for i in range(len(l)-1):
        Path += l[i] + '/'
    Path += logo + str(num) + '.jpg'
    return Path



def processOneImage(inputPath, outputPaths):
    image = io.imread(inputPath)
    greyImage = rgb2grey(image)
    threshold = threshold_otsu(greyImage)
    imgout = closing(greyImage > threshold, square(1))
    imgout = crop(imgout)
    imgout = transform.resize(imgout, (max(imgout.shape), max(imgout.shape)))
    for outputPath in outputPaths:
        io.imsave(outputPath, imgout)


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


if not os.path.isdir("./TrainingSet/"):
    os.mkdir("./TrainingSet/")

logos = ['audi', 'bmw', 'chevrolet', 'honda', 'lexus', 'toyota', 'volkswagon', 'benz']
#logos = ['volkswagon']
for logo in logos:
    num = 1
    for image in glob.glob('./Logos/' + logo + '/*.*'):
        if image.endswith('.jpg') or image.endswith('.jpeg') or image.endswith('.png') or image.endswith('.bmp'):
            inputPath = image
            outputPath1 = getOutputPath(image.split('/'), logo, num)

            outputPath2_dir = "./TrainingSet/" + logo + "/"
            if not os.path.isdir(outputPath2_dir):
                os.mkdir(outputPath2_dir)

            outputPath2 = outputPath2_dir + str(num) + ".jpg"
            num += 1
            processOneImage(image, [outputPath2])
