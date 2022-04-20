# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:07:35 2022

@author: lina3953
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from gen_read import GEN

num_chan = 2
win_size = 504
cor_ind = np.arange(0, 305, 2)

def return_int_label(label):
    if label == 0:
        return np.array([1, 0, 0, 0, 0, 0, 0])
    elif label == 1:
        return np.array([0, 1, 0, 0, 0, 0, 0])
    elif label == 2:
        return np.array([0, 0, 1, 0, 0, 0, 0])
    elif label == 3:
        return np.array([0, 0, 0, 1, 0, 0, 0])
    elif label == 4:
        return np.array([0, 0, 0, 0, 1, 0, 0])
    elif label == 5:
        return np.array([0, 0, 0, 0, 0, 1, 0])
    else:
        return np.array([0, 0, 0, 0, 0, 0, 1])
    
def return_int_label_N2(label):
    if label == 0:
        return 0
    elif label == 1:
        return 0
    elif label == 2:
        return 1
    elif label == 3:
        return 0
    elif label == 4:
        return 0
    elif label == 5:
        return 0
    else:
        return 0

def return_label_vec(arr):
    ret = []
    for i in arr:
        ret.append(return_int_label(i))
    ret = np.array(ret, dtype = np.int32)
    return ret

def return_label_vec_N2(arr):
    ret = []
    for i in arr:
        ret.append(return_int_label_N2(i))
    ret = np.array(ret, dtype = np.int32)
    return ret


model_conv_N2 = tf.keras.Sequential(
    [
        layers.Input(shape=(win_size,1)),
        layers.Conv1D(filters=64, kernel_size=50, padding="same", strides=2, activation="relu"),
        layers.MaxPool1D(),
        layers.Dropout(rate=0.2),
        layers.Conv1D(filters=32, kernel_size=10, padding="same", strides=2, activation="relu"),
        layers.MaxPool1D(),
        #layers.Conv1D(filters=16, kernel_size=7, padding="same", strides=2, activation="relu"),
        #layers.Conv1D(filters=8, kernel_size=7, padding="same", strides=2, activation="relu"),
        #layers.Conv1D(filters=4, kernel_size=7, padding="same", strides=2, activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation = 'relu'),
        layers.Dense(128, activation = 'relu'),
        layers.Dense(1, activation = 'sigmoid')
        ]
)
model_conv.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics = ['accuracy'])
model_conv_N2.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics = ['accuracy'])
#model_conv.summary()

for _ in range(600):
    index = int(np.random.random()*len(cor_ind))
    ind = cor_ind[index]
    gen = GEN(ind)
    epochs, labels = gen.creat_labeled_epochs_single_channel(num_of_epochs=5000)
    int_label = return_label_vec_N2(labels)
    epochs = epochs*10e5
    loss, acc = model_conv_N2.train_on_batch(epochs, int_label)
    #model_conv_N2.fit(epochs, int_label, epochs = 3)
    #print(_)
    #if _%1 == 0:
    print('{} \t{} \t{}'.format(_,loss,acc))
