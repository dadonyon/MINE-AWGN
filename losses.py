import tensorflow as tf
from tensorflow.keras import backend as K
from data_loader import KerasBatchGenerator


class DVContinuousLoss(tf.keras.losses.Loss):
    def __init__(self, name='dv_loss'):
        super(DVContinuousLoss, self).__init__(name=name, reduction='none')

    def call(self, T, T_tild, **kwargs):
        return T_tildLoss().call(T_tild=T_tild) - TLoss().call(T=T)


class DVNumericalLoss(tf.keras.losses.Loss):
    def __init__(self, name='dv_loss'):
        super(DVNumericalLoss, self).__init__(name=name, reduction='none')

    def call(self, t1, t2, **kwargs):

        loss_t = K.mean(t1)
        loss_et = K.log(K.mean(K.exp(t2)))
        return loss_et - loss_t


class TLoss(tf.keras.losses.Loss):
    def __init__(self, name='identity_loss'):
        super(TLoss, self).__init__(name=name, reduction='none')

    def call(self, T, **kwargs):
        mean_T = tf.reduce_mean(T)
        return mean_T


class T_tildLoss(tf.keras.losses.Loss):
    def __init__(self, name='identity_loss'):
        super(T_tildLoss, self).__init__(name=name, reduction='none')

    def call(self, T_tild, **kwargs):
        mean_T_tild = tf.math.log(tf.reduce_mean(tf.math.exp(T_tild)))
        return mean_T_tild

"""
Data = KerasBatchGenerator(std_n=1, std_x=1, dim_y=1, dim_x=1, batch_size=50000)
batch_pxpy = [b for b in Data.generator_PXPY()][0]
batch_pxy = [b for b in Data.generator_PXY()][0]

T = K.log(K.sqrt(tf.constant(2)/tf.constant(1))) + K.pow(batch_pxy[:,1], 2) / 4 - K.pow(batch_pxy[:,0] - batch_pxy[:,1], 2) / 2.
T_tild = K.log(K.sqrt(tf.constant(2)/tf.constant(1))) + K.pow(batch_pxpy[:,1], 2) / 4 - K.pow(batch_pxpy[:,0] - batch_pxpy[:,1], 2) / 2

loss1 = DVNumericalLoss().call(t1=T, t2=T_tild)
loss2 = DVNumericalLoss().call(t1=T, t2=T_tild)
loss = loss1 - loss2
"""