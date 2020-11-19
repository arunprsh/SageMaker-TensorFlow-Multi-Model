from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint 
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import utils
from subprocess import call
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_DATA'))
    parser.add_argument('--output', type=str, default=os.environ.get('SM_CHANNEL_OUTPUT'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    return parser.parse_known_args()


if __name__ == '__main__':
    #call('pip list')
    args, _ = parse_args()
    # Load Data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # Prep Data
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255
    # Split Data
    # one-hot encode the labels
    num_classes = len(np.unique(y_train))
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)
    # break training set into training and validation sets
    X_train, X_validation = X_train[5000:], X_train[:5000]
    y_train, y_validation = y_train[5000:], y_train[:5000]
    # Define Model Architecture
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    # Compile Model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Train Model
    model.fit(X_train, 
              y_train, 
              batch_size=32, 
              epochs=10, 
              validation_data=(X_validation, y_validation), 
              callbacks=[], 
              verbose=2, 
              shuffle=True)
    # Save Model
    model.save(f'{args.model_dir}/1')
