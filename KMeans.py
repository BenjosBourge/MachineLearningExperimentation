import numpy as np
import pygame
from matplotlib import pyplot as plt
from scipy.signal import square
from sklearn.datasets import *
from sklearn.metrics import accuracy_score


def get_distance(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2


def Kmeans(train_X, test_X, train_y, test_y, n, seed):
    yy = []
    centers = []
    centers_value = []

    np.random.seed(seed)

    for _ in range(n):
        centers.append([np.random.uniform(0, 1), np.random.uniform(0, 1)])

    for _ in range(10):
        centers_value = []
        # ranks
        v = []
        for pos in train_X:
            min_dist = float('inf')
            cluster_nb = -1
            for j in range(len(centers)):
                center = centers[j]
                dist = get_distance(pos, center)
                if dist < min_dist:
                    min_dist = dist
                    cluster_nb = j
            v.append(cluster_nb)

        for i in range(len(centers)):
            sum = [0, 0]
            nb = 0
            sum_value = 0
            for j in range(len(train_X)):
                if i == v[j]:
                    sum[0] += train_X[j][0]
                    sum[1] += train_X[j][1]
                    nb += 1
                    sum_value += train_y[j]
            if nb == 0:
                nb = 1
            centers[i] = (sum[0] / nb, sum[1] / nb)
            centers_value.append((sum_value / nb) > 0.5)

    for i in range(len(test_y)):
        min_dist = float('inf')
        cluster_nb = -1
        for j in range(len(centers)):
            center = centers[j]
            dist = get_distance(test_X[i], center)
            if dist < min_dist:
                min_dist = dist
                cluster_nb = j
        yy.append(centers_value[cluster_nb])

    return yy
