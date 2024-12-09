import numpy as np
import pygame
from matplotlib import pyplot as plt
from sklearn.datasets import *
from sklearn.metrics import accuracy_score


from Tree import *


def setup_screen(height, width):
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("NN experimentation")
    return screen

def searched_func(x):
    return x*x - 4

def genetic_func(x, tree):
    return tree.func(x)

def main():
    np.random.seed()

    pygame.init()

    width, height = 960, 640
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    screen = setup_screen(height, width)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        deltaTime = clock.get_time() / 1000

        screen.fill((30, 30, 30))

        for i in range(21):
            width = 1
            color = (120, 120, 120)
            if i == 10:
                width = 2
                color = (160, 160, 160)
            pygame.draw.line(screen, color, (i * 25 + 300, 50), (i * 25 + 300, 550),
                             width=width)
            pygame.draw.line(screen, color, (300, 550 - i * 25), (300 + 500, 550 - i * 25),
                             width=width)

        for i in range(50):
            y0 = searched_func(-10 + 20/50 * i)
            y1 = searched_func(-10 + 20/50 * (i + 1))
            if y0 > 10:
                y0 = 10
            if y0 < -10:
                y0 = -10
            if y1 > 10:
                y1 = 10
            if y1 < -10:
                y1 = -10
            pygame.draw.line(screen, (255, 0, 0), (i * 10 + 300, 300 - y0 * 25), ((i + 1) * 10 + 300, 300 - y1 * 25),
                             width=3)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()
