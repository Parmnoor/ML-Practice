'''
In order to understand all of the functions of Machine Learning, I will create my own ML functions from scratch using this file
'''

import numpy as np

def ReLU(z):
    return np.maximum(0, z)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)