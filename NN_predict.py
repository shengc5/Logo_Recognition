# NN_predict.py
# TianYang Jin, Sheng, Chen
# CSE 415 Project
#

import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from skimage import io, transform, morphology
from skimage.feature import hog
from skimage import data, color, exposure
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import rgb2grey, label2rgb
from skimage.util import pad

def predictImage(img_path, theta_path):
    brands = ["audi", "benz", "bmw", "chevrolet", "honda", "lexus", "toyota", "volkswagon"]
    processOneImage(img_path, './temp.jpg')
    image = io.imread('./temp.jpg')
    image = transform.resize(image, (400, 400))

    features = np.array([hog(image, orientations=8, pixels_per_cell=(20, 20), cells_per_block=(1, 1))])
    thetas = np.transpose(np.load(theta_path))
    os.remove('./temp.jpg')

    prediction = predict(thetas, features)
    return brands[prediction]

def processOneImage(inputPath, outputPath):
    image = io.imread(inputPath)
    greyImage = rgb2grey(image)
    threshold = threshold_otsu(greyImage)
    imgout = closing(greyImage > threshold, square(1))
    imgout = crop(imgout)
    imgout = transform.resize(imgout, (max(imgout.shape), max(imgout.shape)))
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

def predict(final_t, X):
    INPUT_LAYER_SIZE = 3200
    HIDDEN_LAYER_SIZE = 1600
    OUTPUT_LAYER_SIZE = 8
    theta1 = np.reshape(final_t[0:HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1)],
                        (HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE + 1), order='F')
    theta2 = np.reshape(final_t[HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1):],
                    (OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE + 1), order='F')
    m = np.size(X, 0)
    p = np.zeros((m, 1))

    h1 = sigmoid(np.c_[np.ones((m, 1)), X].dot(np.transpose(theta1)))
    h2 = sigmoid(np.c_[np.ones((m, 1)), h1].dot(np.transpose(theta2)))

    p = np.amax(h2, 1)
    dummy = np.argmax(h2, 1)
    return dummy

def sigmoid(z):
    return np.divide(1.0, 1.0 + np.exp(-1 * z))
