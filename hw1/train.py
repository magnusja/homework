#!/usr/bin/env python

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization

import numpy as np

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('obs_file', type=str)
    parser.add_argument('action_file', type=str)
    parser.add_argument('name', type=str)
    args = parser.parse_args()

    observations = np.load(args.obs_file)
    actions = np.load(args.action_file)
    labels = actions.reshape((-1, actions.shape[2]))

    model = Sequential()
    model.add(Dense(units=1024, input_dim=observations.shape[1]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=actions.shape[2]))

    model.compile(loss='mean_squared_error', optimizer='adam', 
              metrics=['mean_squared_error'])

    history = model.fit(observations, labels, batch_size=256, epochs=15)

    model.save('models/' + args.name + '.h5')


if __name__ == '__main__':
    main()
