import tensorflow as tf
from keras.layers import Activation, Dense, BatchNormalization, MaxPool2D, Lambda, Input, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.initializers import he_normal

from .layers import Maxout


#function that return the stuck of Conv2D and MFM
def MaxOutConv2D(x, dim, kernel_size, strides, padding='same'):
    conv_out = Conv2D(dim, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    mfm_out = Maxout(int(dim/2))(conv_out)
    return mfm_out


#function that return the stuck of FC and MFM
def MaxOutDense(x, dim):
    dense_out = Dense(dim)(x)
    mfm_out = Maxout(int(dim/2))(dense_out)
    return mfm_out

# this function helps to build LCNN. 
def build_lcnn(shape, n_label=2):
    """
    Auguments:
     shape (list) : 
      Input shape for LCNN. (Example : [128, 128, 1])
     n_label (int) : 
      Number of label that LCNN should predict.
    """
    
    input = Input(shape=shape)

    conv2d_1 = MaxOutConv2D(input, 64, kernel_size=5, strides=1, padding='same')
    maxpool_1 = MaxPool2D(pool_size=(2, 2), strides=(2,2))(conv2d_1)

    conv_2d_2 = MaxOutConv2D(maxpool_1, 64, kernel_size=1, strides=1, padding='same')
    batch_norm_2 = BatchNormalization()(conv_2d_2)

    conv2d_3 = MaxOutConv2D(batch_norm_2, 96, kernel_size=3, strides=1, padding='same')
    maxpool_3 = MaxPool2D(pool_size=(2, 2), strides=(2,2))(conv2d_3)
    batch_norm_3 = BatchNormalization()(maxpool_3)

    conv_2d_4 = MaxOutConv2D(batch_norm_3, 96, kernel_size=1, strides=1, padding='same')
    batch_norm_4 = BatchNormalization()(conv_2d_4)

    conv2d_5 = MaxOutConv2D(batch_norm_4, 128, kernel_size=3, strides=1, padding='same')
    maxpool_5 = MaxPool2D(pool_size=(2, 2), strides=(2,2))(conv2d_5)

    conv_2d_6 = MaxOutConv2D(maxpool_5, 128, kernel_size=1, strides=1, padding='same')
    batch_norm_6 = BatchNormalization()(conv_2d_6)

    conv_2d_7 = MaxOutConv2D(batch_norm_6, 64, kernel_size=3, strides=1, padding='same')
    batch_norm_7 = BatchNormalization()(conv_2d_7)

    conv_2d_8 = MaxOutConv2D(batch_norm_7, 64, kernel_size=1, strides=1, padding='same')
    batch_norm_8 = BatchNormalization()(conv_2d_8)

    conv_2d_9 = MaxOutConv2D(batch_norm_8, 64, kernel_size=3, strides=1, padding='same')
    maxpool_9 = MaxPool2D(pool_size=(2, 2), strides=(2,2))(conv_2d_9)
    flatten = Flatten()(maxpool_9)

    dense_10 = MaxOutDense(flatten, 160)
    batch_norm_10 = BatchNormalization()(dense_10)
    dropout_10 = Dropout(0.75)(batch_norm_10)

    output = Dense(n_label, activation='softmax')(dropout_10)
            
    return Model(inputs=input, outputs=output)