import numpy as np
import pygame
from matplotlib import pyplot as plt
from scipy.signal import square
from sklearn.datasets import *
from sklearn.metrics import accuracy_score

from KNN import *
from NaivesBayes import *
from DecisionTree import *
from KMeans import *

def setup_screen(height, width):
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("NN experimentation")
    return screen

def get_color_per_result(y, yy):
    color = (255, 255, 255)
    if y == 1:
        if yy == 1:
            color = (0, 255, 0)  # TP = GREEN
        else:
            color = (255, 0, 255)  # FN = PURPLE
    else:
        if yy == 1:
            color = (255, 255, 0)  # FP = YELLOW
        else:
            color = (255, 0, 0)  # TN = RED
    return color

def split_train_test(X, y, ratio):
    r_indexs = np.random.choice(range(0, len(X)), size=int(len(X)*ratio), replace=False).tolist()
    train_X = []
    train_y = []
    test_X = []
    test_y = []
    for i in range(len(X)):
        found = False
        for j in r_indexs:
            if j == i:
                found = True
                break
        if not found:
            test_X.append(X[i])
            test_y.append(y[i])
        else:
            train_X.append(X[i])
            train_y.append(y[i])
    return train_X, test_X, train_y, test_y



def draw_square(screen, xsquare, ysquare, dimensions, square_size, X, y, yy, name):
    np.random.seed()
    font = pygame.font.SysFont(None, 24)
    min_x = dimensions[0]
    min_y = dimensions[1]
    max_x = dimensions[2]
    max_y = dimensions[3]
    pygame.draw.line(screen, (120, 120, 120), (xsquare - 10, ysquare - 10),(xsquare - 10, ysquare + square_size + 10))
    pygame.draw.line(screen, (120, 120, 120), (xsquare + square_size + 10, ysquare - 10),(xsquare + square_size + 10, ysquare + square_size + 10))
    pygame.draw.line(screen, (120, 120, 120), (xsquare - 10, ysquare + square_size + 10),(xsquare + square_size + 10, ysquare + square_size + 10))
    pygame.draw.line(screen, (120, 120, 120), (xsquare + square_size + 10, ysquare - 10),(xsquare - 10, ysquare - 10))

    for i in range(len(X)):
        x = X[i]
        x0 = ((x[0] - min_x) / (max_x - min_x)) * square_size + xsquare
        x1 = ((x[1] - min_y) / (max_y - min_y)) * square_size + ysquare
        icolor = get_color_per_result(y[i], yy[i])
        pygame.draw.circle(screen, icolor, (x0, x1), 4)
    text = font.render(f"{name}", True, (255, 255, 255))
    screen.blit(text, (xsquare + square_size/2 - len(name) * 5, ysquare - 30))


def main():
    np.random.seed()

    pygame.init()

    width, height = 1050, 640
    clock = pygame.time.Clock()
    screen = setup_screen(height, width)
    font = pygame.font.SysFont(None, 24)

    X, y = make_moons(n_samples=200, noise=0.2) #make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    y = y.reshape((y.shape[0], 1))

    train_X, test_X, train_y, test_y = split_train_test(X, y, 0.8)

    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    min_r = 0
    max_r = 0

    for y0 in y:
        min_r = min(min_r, y0)
        max_r = max(max_r, y0 + 1)

    for x in X:
        min_x = min(min_x, x[0])
        max_x = max(max_x, x[0])
        min_y = min(min_y, x[1])
        max_y = max(max_y, x[1])

    dimensions = (min_x, min_y, max_x, max_y)

    running = True
    timer = 0.
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        deltaTime = clock.get_time() / 1000
        timer += deltaTime
        if timer > 2:
            timer = 0.
            train_X, test_X, train_y, test_y = split_train_test(X, y, 0.8)

        screen.fill((50, 50, 50))

        text = font.render(f"TP = GREEN, FN = PURPLE, FP = YELLOW, TN = RED", True, (255, 255, 255))
        screen.blit(text, (300, 10))

        trX = train_X
        teX = test_X
        trY = train_y
        teY = test_y
        if timer < 1.:
            trX = X
            teX = X
            trY = y
            teY = y

        draw_square(screen, 50, 100, dimensions, 200, X, y, y, "Truth")
        draw_square(screen, 400, 100, dimensions, 200, teX, teY, KNN(trX, teX, trY, teY, 3), "KNN")
        draw_square(screen, 750, 100, dimensions, 200, teX, teY, Kmeans(trX, teX, trY, teY), "K-Means")
        draw_square(screen, 50, 400, dimensions, 200, teX, teY, NaiveBayes(trX, teX, trY, teY), "Naives Bayes")
        draw_square(screen, 400, 400, dimensions, 200, teX, teY, DecisionTree(trX, teX, trY, teY), "Decision Tree")
        draw_square(screen, 750, 400, dimensions, 200, test_X, test_y, test_y, "Testing Sample")

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()
