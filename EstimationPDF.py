import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from keras.optimizers import Adam, SGD, RMSprop
from numpy.random import uniform, choice
from numpy import linalg
import time
import math
import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
DIR = "/common_space_docker/Yonatan/Version2MINE/"


def gpu_init():
    """ Allows GPU memory growth """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')


gpu_init()

from data_loader import KerasBatchGenerator
from Functions import FC_mine, ModelCompile, TrainingModels
from trainers import ModelTrainer
from losses import DVContinuousLoss, TLoss


def Estimate_p_y_given_x(x, T1, T2, T3, std_n):
    r = np.linspace(-5, 5, 10000)
    L_estim = []
    L_real = []

    for y in r:
        inp_xy = np.array([x, y]).reshape((-1, 2))
        inp_x = np.array([x]).reshape((-1, 1))
        t1 = float(np.array(T1(inp_xy, training=False)))
        t2 = float(np.array(T2(inp_x, training=False)))
        L_estim.append((math.e ** (t1 - t2)))
        L_real.append((1 / np.sqrt(2 * math.pi * std_n**2)) * math.e ** ((-(y - x) ** 2) / (2 * std_n**2)))

    I = np.trapz(y=L_estim, x=list(r), dx=0.001)
    L_estim = [c / I for c in L_estim]

    plt.plot(r, L_estim, 'b', label="Estimated p(y|x={})".format(x))
    plt.plot(r, L_real, 'r', label="Analytic p(y|x={})".format(x))
    plt.vlines(x=x, ymin=-0.1, ymax=np.max(np.array(L_estim)) + 0.1, colors='k', label='y={}'.format(x))
    plt.xlabel('y')
    plt.title('AWGN p(y|x=1) Estimation'.format(x))
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()


def Estimate_p_x_given_y(y, T1, T2, T3, std_n):
    r = np.linspace(-5, 5, 10000)
    L_estim = []
    L_real = []

    for x in r:
        inp_xy = np.array([x, y]).reshape((-1, 2))
        inp_y = np.array([x]).reshape((-1, 1))
        t1 = float(np.array(T1(inp_xy, training=False)))
        t3 = float(np.array(T3(inp_y, training=False)))
        L_estim.append((math.e ** (t1 - t3)))
        L_real.append((1 / np.sqrt(2 * math.pi * std_n**2)) * math.e ** ((-(x - y) ** 2) / (2 * std_n**2)))

    I = np.trapz(y=L_estim, x=list(r), dx=0.001)
    L_estim = [c / I for c in L_estim]

    plt.plot(r, L_estim, 'b', label="Estimated p(x|y={})".format(y))
    plt.plot(r, L_real, 'r', label="Analytic p(x|y={})".format(y))
    plt.vlines(x=y, ymin=-0.1, ymax=np.max(np.array(L_estim)) + 0.1, colors='k', label='x={}'.format(y))
    plt.xlabel('x')
    plt.title('AWGN p(x|y={})'.format(y))
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def Estimate_p_x(T2, std_x):
    r = np.linspace(-5, 5, 10000)
    L_estim = []
    L_real = []

    for x in r:
        inp_y = np.array([x]).reshape((-1, 1))
        t2 = float(np.array(T2(inp_y, training=False)))
        L_estim.append((math.e ** t2))
        L_real.append((1 / np.sqrt(2 * math.pi * std_x**2)) * math.e ** ((-x ** 2) / (2 * std_x**2)))

    I = np.trapz(y=L_estim, x=list(r), dx=0.001)
    L_estim = [c / I for c in L_estim]

    plt.plot(r, L_estim, 'b', label="Estimated p(x)")
    plt.plot(r, L_real, 'r', label="Analytic p(x)")
    plt.xlabel('x')
    plt.title('AWGN p(x) Estimation')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def Estimate_p_y(T3, std_x, std_n):
    r = np.linspace(-5, 5, 10000)
    L_estim = []
    L_real = []

    for x in r:
        inp_y = np.array([x]).reshape((-1, 1))
        t2 = float(np.array(T3(inp_y, training=False)))
        L_estim.append((math.e ** t2))
        L_real.append((1 / np.sqrt(2 * math.pi * (std_n**2+std_x**2))) * math.e ** ((-x ** 2) / (2 * (std_n**2+std_x**2))))

    I = np.trapz(y=L_estim, x=list(r), dx=0.001)
    L_estim = [c / I for c in L_estim]

    plt.plot(r, L_estim, 'b', label="Estimated p(y)")
    plt.plot(r, L_real, 'r', label="Analytic p(y)")
    plt.xlabel('x')
    plt.title('AWGN p(y) Estimation')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def Estimate_p_xy(T1, T2, T3):
    pass


if __name__ == '__main__':
    T1 = keras.models.load_model(
        "/common_space_docker/Yonatan/Version2MINE/TRAINED_MODELS/AWGN_std_x=1.0,std_n=0.1_3nets/Saved_epoch42_T1/")
    T2 = keras.models.load_model(
        "/common_space_docker/Yonatan/Version2MINE/TRAINED_MODELS/AWGN_std_x=1.0,std_n=0.1_3nets/Saved_epoch42_T2/")
    T3 = keras.models.load_model(
        "/common_space_docker/Yonatan/Version2MINE/TRAINED_MODELS/AWGN_std_x=1.0,std_n=0.1_3nets/Saved_epoch42_T3/")

    # Estimate_p_y_given_x(x=0.55, T1=T1, T2=T2, T3=T3)
    # v Estimate_p_x_given_y(y=-2.12, T1=T1, T2=T2, T3=T3)
    # Estimate_p_y(T3)
