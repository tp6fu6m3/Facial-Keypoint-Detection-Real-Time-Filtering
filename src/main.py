#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import matplotlib.pyplot as plt

from utils import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import ModelCheckpoint   

# Load training set
X_train, y_train = load_data()
X_test, _ = load_data(test=True)

def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Convolution2D(16, kernel_size=2, padding='same', input_shape=(96, 96,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(30, activation='tanh'))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    return model


model = KerasRegressor(build_fn=create_model, epochs=5, batch_size=64, verbose=0)
checkpointer = ModelCheckpoint(filepath='../model/my_model.h5', verbose=1, save_best_only=True)

hist = model.fit(X_train, y_train,
              batch_size=64,
              epochs=20,
              validation_split=0.2,
              callbacks=[checkpointer],
              verbose=2, 
              shuffle=True)

plt.rcParams['figure.figsize'] = (12.0, 9.0)
plt.subplot(211)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.subplot(212)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


y_test = model.predict(X_test)
fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
    plot_data(X_test[i], y_test[i], ax)

