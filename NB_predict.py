# NB_predict.py
# TianYang Jin, Sheng, Chen
# CSE 415 Project
#

import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
from skimage import io, transform
import os
import sys
import math
import random
from skimage.color import rgb2grey, label2rgb
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square


def load_training_set(inputPath):
    training_set = np.load(inputPath)
    features = training_set['arr_0']
    brands = training_set['arr_1']

    return features, brands


def get_probs(features, brands):
    prob_ones = np.zeros((8, 2500))
    prob_zeros = np.zeros((8, 2500))

    for i in range(8):
        car_index = i + 1
        ind = np.where(brands == car_index)[0]
        for j in range(features.shape[1]):
            sum = 0
            for index in ind:
                sum += features[index][j]
            if sum > 0:
                prob_ones[i][j] = sum / len(ind)
            else:
                prob_ones[i][j] = 0.01 / (len(ind) + 0.01)

            if prob_ones[i][j] == 1:
                prob_ones[i][j] = 1 - 0.01 / (len(ind) + 0.01)

            prob_zeros[i][j] = 1 - prob_ones[i][j]
    # for i in range(8):
    #     plt.figure()
    #     plt.imshow(np.reshape(prob_zeros[i, :], (50, 50)))
    # plt.show()
    return prob_ones, prob_zeros


def get_likelihood(prob_ones, prob_zeros, brands, logo, input):
    logos = ['audi', 'bmw', 'chevrolet', 'honda', 'lexus', 'toyota', 'volkswagon', 'benz']
    car_index = logos.index(logo)
    ind = np.where(brands == car_index+1)[0]
    p_bi = len(ind) / len(brands)
    log_P = 0

    for i in range(2500):
        if input[i] == 1:
            log_P += math.log(prob_ones[car_index, i])
        else:
            log_P += math.log(prob_zeros[car_index, i])
    return log_P + math.log(p_bi)


def predict(input_path, data_path):
    logos = ['audi', 'bmw', 'chevrolet', 'honda', 'lexus', 'toyota', 'volkswagon', 'benz']

    # load image and resize
    processOneImage(input_path, './temp.jpg')
    image = io.imread('./temp.jpg')
    small_img = transform.resize(image, (50, 50)) == 1
    flatened_img = np.reshape(small_img, (1, 2500)).astype(np.int)[0]

    # generate feature and calculate probability and max likelihood
    features, brands = load_training_set(data_path)
    prob_ones, prob_zeros = get_probs(features, brands)
    max_likelihood = -sys.maxsize+1
    car_index = -1
    for i in range(len(logos)):
        cur_likelihood = get_likelihood(prob_ones, prob_zeros,brands, logos[i], flatened_img)
        if cur_likelihood > max_likelihood:
            max_likelihood = cur_likelihood
            car_index = i
    os.remove('./temp.jpg')
    return logos[car_index]


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
