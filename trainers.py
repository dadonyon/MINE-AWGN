import os
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import shutil
from keras.optimizers import Adam, SGD, RMSprop
from losses import DVContinuousLoss, TLoss, T_tildLoss, DVNumericalLoss
from data_loader import KerasBatchGenerator

DIR = "/common_space_docker/Yonatan/Version2MINE/"


class ModelTrainer(object):

    def __init__(self, TrainingModels, optimizer, loss, momentum, lr, epochs, batch_size, std_x, std_n, dim_x, dim_y,
                 num_samples,
                 run_description):
        self.TrainingModels = TrainingModels
        self.optimizer = optimizer
        self.momentum = momentum
        self.learning_rate = lr

        self.loss_fn = DVNumericalLoss()

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches = int(num_samples / batch_size + 1)
        self.std_x = std_x
        self.std_n = std_n

        self.num_samples = num_samples
        self.dim_x = dim_x
        self.dim_y = dim_y

        self.run_description = run_description
        self.estim_MI = []

        self.theoretical_MI = 0.5 * math.log(1 + (self.std_x / self.std_n) ** 2)

    @tf.function
    def train_step(self, batch_PXY, batch_UXY, batch_PX, batch_PY, batch_UX, batch_UY):
        """
        Trains the deep neural network with the DV loss. it gets two batches of data:
        Batch_PXY ~ PXY, Batch_PXPY ~ PXPY. Each contains m samples
        """
        x = self.TrainingModels.model1.trainable_weights
        with tf.GradientTape() as tape:
            tape.watch(x)
            T = self.TrainingModels.model1(batch_PXY, training=True)
            T_tild = self.TrainingModels.model1(batch_UXY, training=True)
            loss1 = self.loss_fn.call(t1=T, t2=T_tild)
            del T, T_tild, batch_PXY, batch_UXY
        grads = tape.gradient(loss1, x)
        self.optimizer.apply_gradients(zip(grads, x))  ## updates trainable network params.

        x = self.TrainingModels.model2.trainable_weights
        with tf.GradientTape() as tape:
            tape.watch(x)
            T = self.TrainingModels.model2(batch_PX, training=True)
            T_tild = self.TrainingModels.model2(batch_UX, training=True)
            loss2 = self.loss_fn.call(t1=T, t2=T_tild)
            del T, T_tild, batch_PX, batch_UX
        grads = tape.gradient(loss2, x)
        self.optimizer.apply_gradients(zip(grads, x))  ## updates trainable network params.

        x = self.TrainingModels.model3.trainable_weights
        with tf.GradientTape() as tape:
            tape.watch(x)
            T = self.TrainingModels.model3(batch_PY, training=True)
            T_tild = self.TrainingModels.model3(batch_UY, training=True)
            loss3 = self.loss_fn.call(t1=T, t2=T_tild)
            del T, T_tild, batch_PY, batch_UY
        grads = tape.gradient(loss3, x)
        self.optimizer.apply_gradients(zip(grads, x))  ## updates trainable network params.
        return -(loss1 - loss2 - loss3)

    @tf.function
    def test_step(self):
        """
        Evaluates the M.I using new samples (x,y)~P(X,Y), (x,y)~P(X)P(Y)
        """
        test_data = KerasBatchGenerator(std_x=self.std_x, std_n=self.std_n, dim_x=self.dim_x, dim_y=self.dim_y,
                                        batch_size=self.num_samples)
        batch_test_pxy = [b for b in test_data.generator_PXY()][0]
        batch_test_uxy = [b for b in test_data.generator_UXY()][0]
        T_pxy = self.TrainingModels.model1(batch_test_pxy, training=False)
        T_uxy = self.TrainingModels.model1(batch_test_uxy, training=False)
        del batch_test_pxy
        del batch_test_uxy
        loss1 = self.loss_fn.call(t1=T_pxy, t2=T_uxy)
        del T_pxy
        del T_uxy


        batch_test_px = [b for b in test_data.generator_PX()][0]
        batch_test_ux = [b for b in test_data.generator_UX()][0]
        T_px = self.TrainingModels.model2(batch_test_px, training=False)
        T_ux = self.TrainingModels.model2(batch_test_ux, training=False)
        del batch_test_px
        del batch_test_ux
        loss2 = self.loss_fn.call(t1=T_px, t2=T_ux)
        del T_px
        del T_ux

        batch_test_py = [b for b in test_data.generator_PY()][0]
        batch_test_uy = [b for b in test_data.generator_UY()][0]
        T_py = self.TrainingModels.model3(batch_test_py, training=False)
        T_uy = self.TrainingModels.model3(batch_test_uy, training=False)
        del batch_test_py
        del batch_test_uy
        loss3 = self.loss_fn.call(t1=T_py, t2=T_uy)
        del T_py
        del T_uy


        return -(loss1 - loss2 - loss3)

    def train_model(self):
        saved_epochs = []
        Data = KerasBatchGenerator(std_x=self.std_x, std_n=self.std_n, dim_x=self.dim_x, dim_y=self.dim_y,
                                   batch_size=self.batch_size)
        for epoch in range(self.epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()
            for k in range(self.num_batches):  # Iterate over the batches of the dataset.
                for step, (batch_pxy, batch_uxy, batch_px, batch_py, batch_ux, batch_uy) in enumerate(
                        zip(Data.generator_PXY(), Data.generator_UXY(), Data.generator_PX(), Data.generator_PY(),
                            Data.generator_UX(), Data.generator_UY())):
                    DVloss_value = self.train_step(batch_PXY=batch_pxy, batch_UXY=batch_uxy, batch_PX=batch_px,
                                                   batch_PY=batch_py, batch_UX=batch_ux, batch_UY=batch_uy)
                    # Log every 1 batches.
                    if (k + 1) * (step + 1) % 1 == 0:
                        print(
                            "Estimated I(X;Y) at step %d: %.5f"
                            % ((k + 1) * (step + 1), DVloss_value)
                        )
                        print("Trained on: %d samples" % (((k + 1) * (step + 1)) * self.batch_size))

            estimated_MI = self.test_step()
            self.estim_MI.append(estimated_MI)
            print('Network estimation on entire data: {}'.format(estimated_MI))
            if epoch == 0:
                MI_max = estimated_MI
            if estimated_MI > MI_max:
                MI_max = estimated_MI
                if not os.path.isdir(DIR + 'TRAINED_MODELS/' + self.run_description):
                    os.mkdir(DIR + 'TRAINED_MODELS/' + self.run_description)
                n = 1
                for model in [self.TrainingModels.model1, self.TrainingModels.model2, self.TrainingModels.model3]:
                    model.save(DIR + 'TRAINED_MODELS/' + self.run_description + '/' +
                               'Saved_epoch{}_T{}'.format(epoch, n))
                    n += 1

                saved_epochs.append(epoch)
                if len(saved_epochs) > 1:
                    for i in range(3):
                        shutil.rmtree(
                            DIR + 'TRAINED_MODELS/' + self.run_description + '/' + 'Saved_epoch{}_T{}'.format(
                                saved_epochs[-2], i+1))
                print('*** Estimated M.I improved ---- model saved ***')

            print("Time taken epoch #{}: %.2f secs".format(epoch) % (time.time() - start_time))
            D = 0
            if epoch >= D:
                self.plot_progress(epoch=epoch, delay=D)
                self.save_history()

    def plot_progress(self, epoch, delay):
        if not os.path.isdir(DIR + 'Graphs/' + self.run_description):
            os.mkdir(DIR + 'Graphs/' + self.run_description + '/')
        plt.figure()
        plt.plot(range(epoch + 1 - delay), self.estim_MI[delay:], "-*b", label="Estimated Mutual Information")
        plt.plot(range(epoch + 1 - delay), [self.theoretical_MI for s in range(epoch + 1 - delay)],
                 "-*r", label="Analytic Mutual Information")

        plt.grid(True)
        plt.legend(loc="best")
        plt.xlabel("Epoch")
        plt.savefig(DIR + 'Graphs/' + self.run_description + '/' + '_MI.png')
        print('*** Saved Learning Progress Plots ***')

    def save_history(self):
        if not os.path.isdir(DIR + 'TRAINING_HISTORY/' + self.run_description):
            os.mkdir(DIR + 'TRAINING_HISTORY/' + self.run_description + '/')

        np.save(DIR + 'TRAINING_HISTORY/' + self.run_description + '/estimated_MI' + '.npy',
                np.array(self.estim_MI))
