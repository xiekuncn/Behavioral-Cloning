#!/usr/bin/env python3
# _*_coding=utf8_*_
import os
import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from keras.layers import Dense, Flatten, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt


def train(data_path):
    batch_size = 64
    train_samples, validation_samples = make_samples(data_path)
    train_generator = generator(train_samples, batch_size)
    validation_generator = generator(validation_samples, batch_size)

    ch, row, col = 3, 160, 320  # Trimmed image format

    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

    model.add(Conv2D(24, (5, 5), padding='same', activation='relu', subsample=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(36, (5, 5), padding='same', activation='relu', subsample=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    model.add(Conv2D(48, (5, 5), padding='same', activation='relu', subsample=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
   
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', subsample=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', subsample=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
   
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))git stauts
    
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='tanh'))
    model.summary()

    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator,
                                         steps_per_epoch=len(train_samples),
                                         validation_data=validation_generator,
                                         validation_steps=len(validation_samples),
                                         epochs=3,
                                         verbose=1)
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    model.save("model.h5")


def make_samples(data_path):
    samples = []
    csv_file = os.path.join(data_path + "driving_log.csv")
    skip_line = True
    with open(csv_file) as f:
        reader = csv.reader(f)
        for line in reader:
            if skip_line:
                skip_line = False
                continue
            for image_index in range(3):
                # path = os.path.join(data_path, line[image_index])
                path = line[image_index]
                line[image_index] = ''.join(path.split())
                angle = float(line[3])
                if image_index == 1:
                    angle += 0.229
                elif image_index == 2:
                    angle -= 0.229
                samples.append({"image": line[image_index],
                                "angle": angle,
                                "flip": False})
                samples.append({"image": line[image_index],
                                "angle": angle,
                                "flip": True})

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples


def generator(samples, batch_size=128):
    num_samples = len(samples)

    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample["image"]
                image = cv2.imread(name)
                if image is None or image.shape != (160, 320, 3):
                    continue
                angle = float(batch_sample["angle"])
                if batch_sample["flip"]:
                    images.append(np.fliplr(image))
                    angles.append(-angle)
                else:
                    images.append(image)
                    angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


if __name__ == "__main__":
    train('/home/bill/Pictures/')
