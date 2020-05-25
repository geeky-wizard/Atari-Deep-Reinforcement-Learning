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
        self.logger = CSVLogger(weights[:-11]+"-model_Log.txt" if weights is not None else "History.txt",
                                append=True)
        
        self.discount_factor = discount_factor
        self.batch_size = batch_size

        self.model = Sequential()

        self.model.add(Conv2D(16,
                              8,
                              strides=(4, 4),
                              activation='relu',
                              input_shape=input_dim,
                              data_format='channels_last'))
        
        self.model.add(Conv2D(32,
                              4,
                              strides=(2, 2),
                              activation='relu',
                              data_format='channels_last'))
        
        self.model.add(Flatten())
        
        self.model.add(Dense(256, activation='relu'))
        
        self.model.add(Dense(action_space))

        self.model.compile(optimizer=ADAM_Opt,
                           loss="mean_squared_error", metrics=["accuracy"])

        # self.model.summary()

        if weights is not None:
            print("\n\n\nWeights Found!!!\t", weights)
            self.model.load_weights(weights)
            

    def predict(self, current_state):
        return self.model.predict(np.moveaxis(current_state.astype(np.float64), [1, 2, 3], [3, 1, 2]), batch_size=1)

    def train(self, batch, target_network, _log = False):
        x = []
        y = []
        for experience in batch:
            x.append(experience['current'].astype(np.float64))

            next_state = experience['next_state'].astype(np.float64)
            next_state_pred = target_network.predict(next_state).ravel()
            next_q_value = np.max(next_state_pred)

            target = list(self.predict(experience['current'])[0])
            
            if experience['done']:
                target[experience['action']] = experience['reward']
            else:
                target[experience['action']] = experience['reward'] + self.discount_factor * next_q_value
            y.append(target)

        self.model.fit(np.moveaxis(np.asarray(x).squeeze(axis=1), [1, 2, 3], [3, 1, 2]),
                        np.asarray(y),
                        batch_size=self.batch_size,
                        epochs=1,
                        verbose=0,
                        callbacks=[self.logger] if _log else None)


    def save(self, filepath="model_weights"):
        self.model.save_weights("Assets/Weights/"+filepath)

    def clean(self):
        del self.model
        del self.batch_size
        del self.discount_factor
        gc.collect()
