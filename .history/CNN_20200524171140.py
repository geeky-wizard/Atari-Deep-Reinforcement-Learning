from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.callbacks import CSVLogger

import gc
import numpy as np

from hyperparameters import *

class CNN:
    def __init__(self, input_dim, action_space,
                 discount_factor=DISCOUNT_FACTOR, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, weights=None):
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        # with strategy.scope():
        self.model = Sequential()

        self.model.add(Conv2D(32,
                              8,
                              strides=(4, 4),
                              padding="valid",
                              activation="relu",
                              input_shape=input_dim,
                              data_format="channels_last"))
        self.model.add(Conv2D(64,
                              4,
                              strides=(2, 2),
                              padding="valid",
                              activation="relu",
                              data_format="channels_last"))
        self.model.add(Conv2D(64,
                              3,
                              strides=(1, 1),
                              padding="valid",
                              activation="relu",
                              data_format="channels_last"))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(action_space))


        # self.model.compile(loss="mean_squared_error",
        #                   optimizer=RMSprop(lr=LEARNING_RATE,
        #                                     rho=0.95,
        #                                     epsilon=0.01),
        #                   metrics=["accuracy"])
        
        self.model.compile(loss="mean_squared_error",
                          optimizer=ADAM_Opt,
                          metrics=["accuracy"])

        self.model.summary()

        if weights is not None:
            print("\n\n\nWeights Found!!!\t", weights)
            self.model.load_weights(weights)

    def predict(self, current_state):
        return self.model.predict(np.moveaxis(current_state.astype(np.float64), [1, 2, 3], [3, 1, 2]), batch_size=1)

    def train(self, batch, target_network):
        x = []
        y = []
        for experience in batch:
            x.append(experience['current'].astype(np.float64))

            next_state = experience['next_state'].astype(np.float64)
            next_state_pred = target_network.predict(next_state).ravel()
            next_q_value = np.max(next_state_pred)

            target = list(self.predict(experience['current'])[0])
            
            if experience['done']:
                target[experience['action']-1] = experience['reward']
            else:
                target[experience['action']-1] = experience['reward'] + self.discount_factor * next_q_value
            y.append(target)

        fit = self.model.fit(np.moveaxis(np.asarray(x).squeeze(axis=1), [1, 2, 3], [3, 1, 2]),
                        np.asarray(y),
                        batch_size=self.batch_size,
                        epochs=1,
                        verbose=0)
        
        loss = fit.history["loss"][0]
        accuracy = fit.history["acc"][0]
        return loss, accuracy

    def save(self, filepath="model.h5"):
        self.model.save_weights(filepath)