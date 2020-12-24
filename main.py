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


def main(batch, epochs, lr, momentum, std_x, std_n):
    optimizer = Adam(lr=lr)
    loss = DVContinuousLoss()
    T1 = FC_mine(dim_x=1, dim_y=1) ## Trained on (X,Y) ~ PXY
    T2 = FC_mine(dim_x=1, dim_y=0) ## Trained on X ~ PX, UX
    T3 = FC_mine(dim_x=0, dim_y=1) ## Trained on Y ~ PY, UY

    TrainModels = TrainingModels(models=[T1, T2, T3])

    Trainer = ModelTrainer(TrainingModels=TrainModels, optimizer=optimizer, loss=loss, momentum=momentum, lr=lr, epochs=epochs, batch_size=batch,
                           std_x=std_x, std_n=std_n, dim_x=1, dim_y=1, num_samples=500000, run_description='AWGN_std_x={},std_n={}_3nets'.format(std_x, std_n))

    Trainer.train_model()


if __name__ == '__main__':
    """
    Data = KerasBatchGenerator(std_n=1, std_x=1, dim_y=1, dim_x=1, batch_size=5000)

    batch_pxy = [b for b in Data.generator_PXY()][0]
    batch_uxy = [b for b in Data.generator_UXY()][0]

    batch_px = [b for b in Data.generator_PX()][0]
    batch_ux = [b for b in Data.generator_UX()][0]

    batch_py = [b for b in Data.generator_PY()][0]
    batch_uy = [b for b in Data.generator_UY()][0]
    """

    main(batch=100000, epochs=60, momentum=0.93, lr=0.001, std_x=1.0, std_n=2)










