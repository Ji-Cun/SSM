# -*- coding: utf-8 -*-

from tensorflow import keras
import pandas as pd
import numpy as np
from src.classifiers.TimeHistory import TimeHistory
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau


class FCN:
    def __init__(self, input_shape, nb_classes):
        self.model = self.build_model(input_shape, nb_classes)
    def build_model(self, input_shape, nb_classes):
        input = keras.layers.Input(input_shape)
        conv1 = keras.layers.Conv1D(128, 8, padding="same")(input)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation('relu')(conv1)

        conv2 = keras.layers.Conv1D(256, 5, padding="same")(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, 3, padding="same")(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        full = keras.layers.GlobalAveragePooling1D()(conv3)
        out = keras.layers.Dense(nb_classes, activation='softmax')(full)

        model = keras.models.Model(inputs=input, outputs=out)

        # optimizer = keras.optimizers.Adam()
        optimizer = keras.optimizers.Nadam()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model
    def fit(self, x_train, x_test, Y_train, Y_test, nb_epochs=1000):
        batch_size = int(min(x_train.shape[0] / 10, 16))

        es = EarlyStopping(monitor='loss', min_delta=0.0001, patience=50)
        rp = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=50, min_lr=0.0001)
        hist = self.model.fit( x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
                              verbose=0, validation_data=(x_test, Y_test), callbacks=rp)
        log = pd.DataFrame(hist.history)
        acc = log.iloc[-1]['val_accuracy']
        return acc


    def fitTimeLog(self, x_train, x_test, Y_train, Y_test, nb_epochs=2000):
        batch_size = int(min(x_train.shape[0] / 10, 16))

        time_callback = TimeHistory()
        hist = self.model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
                              verbose=0, validation_data=(x_test, Y_test), callbacks=[time_callback])
        log = pd.DataFrame(hist.history)
        df_time = pd.DataFrame(time_callback.times)
        return df_time

    def predict(self, x_test):
        y_pred = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
