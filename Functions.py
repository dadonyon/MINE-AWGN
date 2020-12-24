import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Add
from tensorflow.keras.layers import Dense, LeakyReLU, Concatenate
from keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Dropout, BatchNormalization
from numpy.random import uniform, choice
from losses import DVContinuousLoss, TLoss


def gpu_init():
    """ Allows GPU memory growth """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')


def FC_mine(dim_x, dim_y):
    I = keras.Input(shape=(dim_x + dim_y,))
    G = I
    for units in [32, 64, 128, 256, 512]:
        G = Dense(units=units, activation='elu', use_bias=True)(G)

    T = Dense(units=1, activation='elu', use_bias=True)(G)

    model = keras.Model(inputs=I, outputs=T, name="Fully_Connected_Network")
    return model


def FC_mine1(dim_x, dim_y):
    I = keras.Input(shape=(dim_x + dim_y,))
    G = I
    for units, DO in zip([32, 64, 128, 256, 512, 1024], [0.1, 0.2, 0.5, 0.6, 0.7]):
        G = Dense(units=units, activation='elu', use_bias=True)(G)
        G = Dropout(rate=DO)(G)

    T = Dense(units=1, activation='elu', use_bias=True)(G)

    model = keras.Model(inputs=I, outputs=T, name="Fully_Connected_Network")
    return model


def ModelCompile(Net, optimizer, loss):
    Net.compile(optimizer=optimizer, loss=loss)
    tf.keras.utils.plot_model(Net,
                              to_file='/common_space_docker/Yonatan/MINE/Model_Graphs/' + 'FullyConnectedNetwork.png',
                              show_shapes=True)
    Net.summary()
    return Net


class TrainingModels(object):
    def __init__(self, models):
        self.model1 = models[0]
        self.model2 = models[1]
        self.model3 = models[2]
