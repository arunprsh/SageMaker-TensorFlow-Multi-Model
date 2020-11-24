from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint 
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras import utils
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import os


# Set Log Level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



def parse_args():
    parser = argparse.ArgumentParser() 
    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    
    parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_DATA'))
    parser.add_argument('--output', type=str, default=os.environ.get('SM_CHANNEL_OUTPUT'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    return parser.parse_known_args()


def get_train_data(train_dir):
    X_train = np.load(os.path.join(train_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(train_dir, 'y_train.npy'))
    print('X train', X_train.shape, 'y train', y_train.shape)
    return X_train, y_train


def get_validation_data(val_dir):
    X_validation = np.load(os.path.join(val_dir, 'X_validation.npy'))
    y_validation = np.load(os.path.join(val_dir, 'y_validation.npy'))
    print('X validation', X_validation.shape, 'y validation', y_validation.shape)
    return X_validation, y_validation


if __name__ == '__main__':
    print(f'Using TensorFlow version: {tf.__version__}')
    DEVICE = '/cpu:0'
    args, _ = parse_args()
    epochs = args.epochs
    # Load Data
    X_train, y_train = get_train_data(args.train)
    X_validation, y_validation = get_validation_data(args.val)
    with tf.device(DEVICE):
        # Data Augmentation
        TRAIN_BATCH_SIZE = 32
        data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        train_iterator = data_generator.flow(X_train, y_train, batch_size=TRAIN_BATCH_SIZE)
        # Define Model Architecture
        model = Sequential()
        # CONVOLUTIONAL LAYER 1
        model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(32, 32, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=2))

        # CONVOLUTIONAL LAYER 1
        model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=2))

        # CONVOLUTIONAL LAYER 3
        model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.3))

        # FULLY CONNECTED LAYER 
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(10, activation='softmax'))
        model.summary()
        # Compile Model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Train Model
        BATCH_SIZE = 32
        STEPS_PER_EPOCH = int(X_train.shape[0]/TRAIN_BATCH_SIZE)
        
        model.fit(train_iterator, 
                  steps_per_epoch=STEPS_PER_EPOCH, 
                  batch_size=BATCH_SIZE, 
                  epochs=epochs, 
                  validation_data=(X_validation, y_validation), 
                  callbacks=[], 
                  verbose=2, 
                  shuffle=True)
        # Save Model
        model.save(f'{args.model_dir}/1')