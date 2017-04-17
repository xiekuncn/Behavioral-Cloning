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
import tensorflow as tf
from keras.models import Sequential
import matplotlib.pyplot as plt
import keras


def train(data_paths):
    '''
    train the model by the data providing from data_paths.
     
    :param data_paths: it's a list of dataset folders.
    
    the code will load  the 'driving_log.csv' for each input folder,
    and this csv file should record the image by relative path, *NOT absolute path*.
     
    If the image is not exist, will raise a FileNotFound exception before training.
    '''
    batch_size = 128
    if type(data_paths) is str:
        data_paths = [data_paths]
    train_samples, validation_samples = [], []
    for data_path in data_paths:
        train_data, validation_data = make_samples(data_path)
        train_samples.extend(train_data)
        validation_samples.extend(validation_data)

    train_generator = generator(train_samples, batch_size)
    validation_generator = generator(validation_samples, batch_size)

    ch, row, col = 3, 160, 320  # Trimmed image format

    model = Sequential()
    model.add(Cropping2D(cropping=((40, 25), (0, 0)), input_shape=(row, col, ch)))
    model.add(Lambda(lambda x: keras.layers.core.K.tf.image.resize_images(x, (66, 200))))  # resize image
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(66, 200, 3),
                     output_shape=(66, 200, 3)))
    model.add(Conv2D(kernel_size=(5, 5), filters=24, padding='valid', activation='relu', strides=(2, 2), use_bias=True))
    model.add(Conv2D(kernel_size=(5, 5), filters=36, padding='valid', activation='relu', strides=(2, 2), use_bias=True))
    model.add(Conv2D(kernel_size=(5, 5), filters=48, padding='valid', activation='relu', strides=(2, 2), use_bias=True))
    model.add(Conv2D(kernel_size=(3, 3), filters=64, padding='valid', activation='relu', strides=(1, 1), use_bias=True))
    model.add(Conv2D(kernel_size=(3, 3), filters=64, padding='valid', activation='relu', strides=(1, 1), use_bias=True))
   
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='tanh'))
    model.summary()

    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator,
                                         steps_per_epoch=len(train_samples),
                                         validation_data=validation_generator,
                                         validation_steps=len(validation_samples),
                                         epochs=3)
    model.save("model.h5")
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    # plt.plot(history_object.history['loss'])
    # plt.plot(history_object.history['val_loss'])
    # plt.title('model mean squared error loss')
    # plt.ylabel('mean squared error loss')
    # plt.xlabel('epoch')
    # plt.legend(['training set', 'validation set'], loc='upper right')
    # plt.show()




def make_samples(data_path):
    '''
    make samples for data_path folder.
    and will return the training samples and validate sample by 4:1.
    
    the data will augmentation by flip, it's a attribute in one sample.
    
    :param data_path: it's a folder includes 'driving_log.csv' file. 
    :return training_samples, validate_samples.
    '''
    samples = []
    csv_file = os.path.join(data_path, "driving_log.csv")
    skip_line = True
    with open(csv_file) as f:
        reader = csv.reader(f)
        for line in reader:
            if skip_line:
                skip_line = False
                continue
            for image_index in range(3):
                path = "".join(line[image_index].split())
                path = os.path.join(data_path, path)
                if not os.path.exists(path):
                    raise FileNotFoundError(path)
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
    '''
    it's a generator for sampling.
    
    :param samples: the whole datasets for training or validation 
    :param batch_size: batch size
    :return: yield a batch sample
    '''
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
    train(['/dataset/pj3-1/', '/dataset/pj3-2/', '/dataset/pj3-origin/'])
