import numpy as np 
import os
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras import backend as K


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def create_unet(input_size):
    
    input_layer = Input(input_size)
    
    conv_1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    conv_1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_1_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1_2)

    conv_2_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_1)
    conv_2_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_2_1)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2_2)

    conv_3_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_2)
    conv_3_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_3_1)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3_2)

    conv_4_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_3)
    conv_4_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_4_1)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(conv_4_2)

    conv_5_1 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
    conv_5_2 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv_5_1)

    up_6 = UpSampling2D(size=2)(conv_5_2)
    up_conv_6 = Conv2D(256, 2, activation='relu', padding='same')(up_6)
    merge_6 = concatenate([up_conv_6, conv_4_2], axis=3)
    conv_6_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge_6)
    conv_6_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_6_1)

    up_7 = UpSampling2D(size=2)(conv_6_2)
    up_conv_7 = Conv2D(128, 2, activation='relu', padding='same')(up_7)
    merge_7 = concatenate([up_conv_7, conv_3_2], axis=3)
    conv_7_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge_7)
    conv_7_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_7_1)

    up_8 = UpSampling2D(size=2)(conv_7_2)
    up_conv_8 = Conv2D(64, 2, activation='relu', padding='same')(up_8)
    merge_8 = concatenate([up_conv_8, conv_2_2], axis=3)
    conv_8_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge_8)
    conv_8_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_8_1)

    up_9 = UpSampling2D(size=2)(conv_8_2)
    up_conv_9 = Conv2D(32, 2, activation='relu', padding='same')(up_9)
    merge_9 = concatenate([up_conv_9, conv_1_2], axis=3)
    conv_9_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge_9)
    conv_9_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_9_1)

    output_layer = Conv2D(1, (1, 1), activation='sigmoid')(conv_9_2)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    
    opt = Adam(lr=0.00001)
    
    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[dice_coef])
    #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model
