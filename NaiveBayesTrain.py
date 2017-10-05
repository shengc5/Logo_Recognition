# NaiveBayesTrain.py
# TianYang Jin, Sheng, Chen
# CSE 415 Project
#

import numpy as np

def get_prior_attr(training_set):
    '''Calcuate and return the priors of attributes values equal to 1.'''
    count = np.zeros((1, 2500))
    m = np.size(training_set, 0)
    for i in range(m):
        for j in range(2500):
            count[j] = training_set



    count = [0] * 2500
    total = len(training_set)
    for training_list in training_set:
        for i in range(2500):
            count[i] += training_list[0][i]
    # avoid probability of 0
    for i in range(2500):
        if count[i] == 0:
            count[i] += 0.01
            total += 0.01
    return [c / total for c in count]


def get_indi(training_set):
    '''Calculate and return the probabilty of individual features.
       [[P(block1|'a')...P(block900|'a')], 'a'], ....]'''
    patterns = "0123456789abcdefghijklmnpqrstuvwxyz"
    indi_prob = []
    for c in patterns:
        count_char = 0
        count_attr_list = [0] * 900
        for training_list in training_set:
            if training_list[1] == c:
                count_char += 1
                for i in range(900):
                    count_attr_list[i] += training_list[0][i]

        for i in range(900):
            if count_attr_list[i] == 0:
                count_attr_list[i] += 0.01
                count_char += 0.01

        indi_prob.append([[count / count_char for count in count_attr_list], c])

    return indi_prob


def get_likelihood(input_list, c, indi_prob, prior_char):
    '''Calculate and return P(c|input)'''
    patterns = "0123456789abcdefghijklmnpqrstuvwxyz"
    char_index = patterns.index(c)
    log_p_input_c = 0
    prob_list = indi_prob[char_index][0]

    for i in range(900):
        if input_list[i] == 0:
            log_p_input_c += math.log(1 - prob_list[i])
        else:
            log_p_input_c += math.log(prob_list[i])
    p_c = prior_char[char_index]
    return log_p_input_c + math.log(p_c)


def get_result(imgpath, indi_prob, prior_char):
    '''Use Naive Bayes classifier to compute
       the character with max likihood of given image array.'''
    input = img_to_array(imgpath)
    max_log_p = -sys.maxsize - 1
    max_c = ''
    for c in "0123456789abcdefghijklmnpqrstuvwxyz":
        log_p = get_likelihood(input, c, indi_prob, prior_char)
        if log_p > max_log_p:
            max_log_p = log_p
            max_c = c
    return (max_c, max_log_p)

data = np.load("nb_data.npz")
X = data['arr_0']
y = data['arr_1']
print(get_prior_attr(X))