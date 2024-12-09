import numpy as np
import pygame
from matplotlib import pyplot as plt
from scipy.signal import square
from sklearn.datasets import *
from sklearn.metrics import accuracy_score

def get_distance(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2


def KNN(train_X, test_X, train_y, test_y, n):
    yy = []

    for i in range(len(test_y)):
        results = []

        for j in range(len(train_y)):
            results.append((train_y[j], get_distance(test_X[i], train_X[j])))
        results = sorted(results, key=lambda x: x[1])
        r = 0
        for _ in range(n):
            r += results[0][0][0]
        r /= n
        if r >= 0.5:
            yy.append(1)
        else:
            yy.append(0)
    return yy
