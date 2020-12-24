import collections
import os
import tensorflow as tf
import numpy as np
from numpy.random import randn, shuffle
import argparse

DIR = "/common_space_docker/Yonatan/Version2MINE/"


# Iterator that will be passed to fit and chunk up our data
class KerasBatchGenerator(object):
    """
    Creates a class that can generates batches sampled from AWGN continuous channel
    """

    def __init__(self, std_x, std_n, dim_x, dim_y, batch_size):
        self.batch_size = batch_size

        self.std_x = std_x
        self.std_n = std_n

        self.dim_x = dim_x
        self.dim_y = dim_y

        self.min_XY = [0, 0]
        self.max_XY = [0, 0]


    def generator_PXY(self):
        """
        return: this generator returns m-tuples (xi,yi) ~ P(X,Y)
        in the respective batch size batch x--> m x (dim_x dim_y)

        """
        batch = []
        for m in range(self.batch_size):
            x = self.std_x * randn(self.dim_x)
            n = self.std_n * randn(self.dim_y)
            y = x + n
            batch.append(np.array([x, y]))

        batch = np.reshape(np.array(batch), (self.batch_size, self.dim_x + self.dim_y))
        self.min_XY = [np.min(batch[:, 0]), np.min(batch[:, 1])]
        self.max_XY = [np.max(batch[:, 0]), np.max(batch[:, 1])]

        yield batch

    def generator_PX(self):
        batch = []
        for m in range(self.batch_size):
            x = self.std_x * randn(self.dim_x)
            batch.append(x)
        yield np.reshape(np.array(batch), (self.batch_size, self.dim_x))

    def generator_PY(self):
        batch = []
        for m in range(self.batch_size):
            x = self.std_x * randn(self.dim_x)
            n = self.std_n * randn(self.dim_y)
            y = x + n
            batch.append(y)
        yield np.reshape(np.array(batch), (self.batch_size, self.dim_y))


    def generator_UXY(self):
        batch = np.random.uniform(low=self.min_XY, high=self.max_XY, size=(self.batch_size, self.dim_x + self.dim_y))
        yield np.reshape(batch, (self.batch_size, self.dim_x + self.dim_y))


    def generator_UX(self):
        batch = np.random.uniform(low=self.min_XY[0], high=self.max_XY[0], size=(self.batch_size, self.dim_x))
        yield np.reshape(batch, (self.batch_size, self.dim_x))

    def generator_UY(self):
        batch = np.random.uniform(low=self.min_XY[1], high=self.max_XY[1], size=(self.batch_size, self.dim_y))
        yield np.reshape(batch, (self.batch_size, self.dim_y))


















