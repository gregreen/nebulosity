#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
from keras.preprocessing import image as kpimage
import keras.models as kmodels
import keras.layers as klayers
import keras.backend as kbackend
import keras.applications as kapplications
import keras.optimizers as koptimizers
import keras.regularizers as kregularizers
import keras.callbacks as kcallbacks
import keras.backend as K

from keras_diagram import ascii as model_to_ascii

import os
import json


# Training/validation data properties
data_dir = os.environ['DATADIR']
project_dir = os.path.expanduser('~/projects/nebulosity')

train_data_dir = os.path.join(data_dir, 'train/')
validation_data_dir = os.path.join(data_dir, 'validation/')
dataflow_dir = os.path.join(data_dir, 'dataflow/')
weight_dir = os.path.join(project_dir, 'weights/')
log_dir = os.path.join(project_dir, 'log/')
validation_res_dir = os.path.join(project_dir, 'validation_res/')
train_res_dir = os.path.join(project_dir, 'train_res/')
checkpoint_dir = os.path.join(project_dir, 'checkpoints/')

train_samples = 2 * 1207
validation_samples = 2 * 300


# Image properties
img_width, img_height = (512, 512) #(512, 512)
if kbackend.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)


# Training settings
epochs = 150
batch_size = 32
output_name = '16th_try'


def add_noise(sigma_frac=0.25, sigma_min=4., floor_max=5.):
    def f(img):
        # Mean and median of all pixels below 25th percentile
        threshold = np.nanpercentile(np.abs(img), 25.)
        idx = np.abs(img) < threshold
        sigma = np.sqrt(np.nanvar(img[idx]))
        mu = np.nanmedian(img[idx])

        # Clip all pixels more than 5 sigma, and recalculate sigma
        idx2 = np.abs(img[idx] - mu) / sigma < 5.
        sigma = np.sqrt(np.nanvar(img[idx][idx2]))
        if not np.isfinite(sigma) or sigma < 1.e-10:
            sigma = sigma_min
        # print(sigma)

        # Add Gaussian noise
        res = img + np.random.normal(size=img.shape, scale=sigma*sigma_frac)
        # Add floor
        res += np.random.random() * floor_max
        # Clip
        res = np.clip(res, 0., 255.)

        return res

    return f


def create_network_v1():
    # Define the neural network architecture
    model = kmodels.Sequential()

    model.add(klayers.Conv2D(
        32, (3, 3),
        input_shape=input_shape,
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        32, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        32, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        32, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Flatten())
    model.add(klayers.Dense(
        64,
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.Dropout(0.5))
    model.add(klayers.Dense(
        1,
        activation='sigmoid',
        kernel_regularizer=kregularizers.l2(0.01)
    ))

    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

    return model


def create_network_v2():
    # Define the neural network architecture
    model = kmodels.Sequential()

    model.add(klayers.Conv2D(
        32, (3, 3),
        input_shape=input_shape,
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        32, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        32, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        48, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        64, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(klayers.Conv2D(
        128, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(klayers.AveragePooling2D(pool_size=(6, 6)))

    model.add(klayers.Flatten())
    #model.add(klayers.Dense(
    #    16,
    #    activation='relu',
    #    kernel_regularizer=kregularizers.l2(0.01)))
    #model.add(klayers.Dropout(0.5))
    model.add(klayers.Dense(
        1,
        activation='sigmoid',
        kernel_regularizer=kregularizers.l2(0.01)
    ))

    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

    return model


def get_class(y_pred):
    return K.round(y_pred)


def count_true(x):
    return K.cast(K.sum(K.cast(x, dtype='int64')), dtype='float64')


def purity_metric(class_index):
    def purity(y_true, y_pred):
        idx_true = K.equal(y_true, class_index)
        idx_pred = K.equal(get_class(y_pred), class_index)
        #print('# predicted to be class {}: {}'.format(class_index, count_true(idx_pred)))
        return count_true(idx_true & idx_pred) / (count_true(idx_pred) + K.epsilon())
    return purity


def completeness_metric(class_index):
    def completeness(y_true, y_pred):
        idx_true = K.equal(y_true, class_index)
        idx_pred = K.equal(get_class(y_pred), class_index)
        return count_true(idx_true & idx_pred) / (count_true(idx_true) + K.epsilon())
    return completeness


def create_network_v3():
    # Define the neural network architecture
    model = kmodels.Sequential()

    model.add(klayers.Conv2D(
        32, (3, 3),
        input_shape=input_shape,
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        38, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        44, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        50, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    #model.add(klayers.Conv2D(
    #    5, (1, 1),
    #    activation='relu',
    #    kernel_regularizer=kregularizers.l2(0.01)))
    #model.add(klayers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(klayers.AveragePooling2D(pool_size=(5, 5)))

    model.add(klayers.Flatten())
    model.add(klayers.Dropout(0.2))
    model.add(klayers.Dense(
        8,
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01),
        bias_regularizer=kregularizers.l2(0.01)
    ))
    model.add(klayers.Dropout(0.1))
    model.add(klayers.Dense(
        1,
        activation='sigmoid',
        kernel_regularizer=kregularizers.l2(0.01),
        bias_regularizer=kregularizers.l2(0.01)
    ))

    model.compile(
        loss='binary_crossentropy',
        optimizer=koptimizers.Adadelta(lr=0.7),#'adadelta',
        metrics=['accuracy', completeness_metric(0), purity_metric(0)])

    return model


def create_network_v4():
    # Define the neural network architecture
    model = kmodels.Sequential()

    model.add(klayers.Conv2D(
        32, (5, 5),
        input_shape=input_shape,
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        24, (5, 5),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        24, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        24, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        24, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(klayers.Conv2D(
        24, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))
    
    #model.add(klayers.Conv2D(
    #    5, (1, 1),
    #    activation='relu',
    #    kernel_regularizer=kregularizers.l2(0.01)))
    #model.add(klayers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(klayers.GlobalAveragePooling2D())

    #model.add(klayers.Flatten())
    model.add(klayers.Dropout(0.2))
    model.add(klayers.Dense(
        8,
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01),
        bias_regularizer=kregularizers.l2(0.01)
    ))
    model.add(klayers.Dropout(0.1))
    model.add(klayers.Dense(
        1,
        activation='sigmoid',
        kernel_regularizer=kregularizers.l2(0.01),
        bias_regularizer=kregularizers.l2(0.01)
    ))

    model.compile(
        loss='binary_crossentropy',
        optimizer=koptimizers.Adadelta(lr=0.7),#'adadelta',
        metrics=['accuracy', completeness_metric(0), purity_metric(0)])

    return model


def create_network_v5():
    # Define the neural network architecture
    model = kmodels.Sequential()

    model.add(klayers.Conv2D(
        24, (3, 3),
        input_shape=input_shape,
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.Conv2D(
        24, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        24, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.Conv2D(
        24, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(klayers.AveragePooling2D(pool_size=(25, 25)))

    model.add(klayers.Flatten())
    model.add(klayers.Dropout(0.2))
    model.add(klayers.Dense(
        8,
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01),
        bias_regularizer=kregularizers.l2(0.01)
    ))
    model.add(klayers.Dropout(0.1))
    model.add(klayers.Dense(
        1,
        activation='sigmoid',
        kernel_regularizer=kregularizers.l2(0.01),
        bias_regularizer=kregularizers.l2(0.01)
    ))

    model.compile(
        loss='binary_crossentropy',
        optimizer=koptimizers.Adadelta(lr=0.7),#'adadelta',
        metrics=['accuracy', completeness_metric(0), purity_metric(0)])

    return model


def create_network_v6():
    # Define the neural network architecture
    model = kmodels.Sequential()

    model.add(klayers.Conv2D(
        12, (3, 3),
        input_shape=input_shape,
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.Conv2D(
        12, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        12, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.Conv2D(
        12, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        12, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.Conv2D(
        12, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(0.01)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.GlobalAveragePooling2D())

    #model.add(klayers.Flatten())
    model.add(klayers.Dropout(0.2))
    model.add(klayers.Dense(
        8,
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.01),
        bias_regularizer=kregularizers.l2(0.01)
    ))
    model.add(klayers.Dropout(0.1))
    model.add(klayers.Dense(
        1,
        activation='sigmoid',
        kernel_regularizer=kregularizers.l2(0.01),
        bias_regularizer=kregularizers.l2(0.01)
    ))

    model.compile(
        loss='binary_crossentropy',
        optimizer=koptimizers.Adadelta(lr=0.05),#'adadelta',
        metrics=['accuracy', completeness_metric(0), purity_metric(0)])

    return model


def create_network_v7():
    # Define the neural network architecture
    model = kmodels.Sequential()

    model.add(klayers.Conv2D(
        12, (5, 5),
        input_shape=input_shape,
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.001)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        24, (5, 5),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.001)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        24, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.001)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        24, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.001)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        24, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.001)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(klayers.Conv2D(
        24, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.001)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))
    
    #model.add(klayers.Conv2D(
    #    5, (1, 1),
    #    activation='relu',
    #    kernel_regularizer=kregularizers.l2(0.01)))
    #model.add(klayers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(klayers.GlobalAveragePooling2D())

    #model.add(klayers.Flatten())
    model.add(klayers.Dropout(0.2))
    model.add(klayers.Dense(
        8,
        activation='relu',
        kernel_regularizer=kregularizers.l2(0.001),
        bias_regularizer=kregularizers.l2(0.001)
    ))
    model.add(klayers.Dropout(0.1))
    model.add(klayers.Dense(
        1,
        activation='sigmoid',
        kernel_regularizer=kregularizers.l2(0.001),
        bias_regularizer=kregularizers.l2(0.001)
    ))

    model.compile(
        loss='binary_crossentropy',
        optimizer=koptimizers.Adadelta(lr=0.7),#'adadelta',
        metrics=['accuracy', completeness_metric(0), purity_metric(0)])

    return model


def create_network_v8():
    # Define the neural network architecture
    model = kmodels.Sequential()
    
    penalty = 0.0001

    model.add(klayers.Conv2D(
        12, (5, 5),
        input_shape=input_shape,
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        24, (5, 5),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        24, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        24, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        24, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(klayers.Conv2D(
        24, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))
    
    #model.add(klayers.Conv2D(
    #    5, (1, 1),
    #    activation='relu',
    #    kernel_regularizer=kregularizers.l2(0.01)))
    #model.add(klayers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(klayers.GlobalAveragePooling2D())

    #model.add(klayers.Flatten())
    model.add(klayers.Dropout(0.2))
    model.add(klayers.Dense(
        8,
        activation='relu',
        kernel_regularizer=kregularizers.l2(penalty),
        bias_regularizer=kregularizers.l2(penalty)
    ))
    model.add(klayers.Dropout(0.1))
    model.add(klayers.Dense(
        1,
        activation='sigmoid',
        kernel_regularizer=kregularizers.l2(penalty),
        bias_regularizer=kregularizers.l2(penalty)
    ))

    model.compile(
        loss='binary_crossentropy',
        optimizer=koptimizers.Adadelta(lr=0.7),#'adadelta',
        metrics=['accuracy', completeness_metric(0), purity_metric(0)])

    return model


def train_full_network():
    """
    Train a full network from scratch.
    """

    model = create_network_v8()
    model.summary()
    print('')
    print(model_to_ascii(model))
    print('')

    train_datagen = kpimage.ImageDataGenerator(
        rescale=1./255.,
        horizontal_flip=True,
        vertical_flip=True)#,
        #preprocessing_function=add_noise())

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        color_mode='grayscale',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        #save_to_dir=dataflow_dir,
        class_mode='binary')

    validation_datagen = kpimage.ImageDataGenerator(
        rescale=1./255.)

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        color_mode='grayscale',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    
    checkpoint_fname = os.path.join(
        checkpoint_dir,
        output_name + '.epoch{epoch:03d}-loss{val_loss:0.3f}.h5')
    
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size,
        callbacks=[
            kcallbacks.CSVLogger(os.path.join(log_dir, output_name + '.log')),
            kcallbacks.EarlyStopping(monitor='val_loss', min_delta=0.005, patience=15),
            kcallbacks.ModelCheckpoint(checkpoint_fname)
        ])
    
    with open(os.path.join(log_dir, output_name + '_model_diagram.log'), 'w') as f:
        f.write(model_to_ascii(model))
    
    with open(os.path.join(weight_dir, output_name + '.json'), 'w') as f:
        f.write(model.to_json(indent=2))
    
    model.save_weights(os.path.join(weight_dir, output_name + '.h5'))


def export_validation_results():
    #model = create_network_v3()
    with open(os.path.join(weight_dir, output_name + '.json')) as f:
        model_json = f.read()
    model = kmodels.model_from_json(model_json)
    model.load_weights(os.path.join(weight_dir, output_name + '.h5'))
    #model = kmodels.load_model(os.path.join(weight_dir, 'first_try.h5'))
    print(model.summary())
    
    tmp_batch_size = 2

    # Compute the output for each validation
    validation_datagen = kpimage.ImageDataGenerator(
        rescale=1./255.)

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        color_mode='grayscale',
        target_size=(img_width, img_height),
        batch_size=tmp_batch_size,
        class_mode='binary',
        shuffle=False,
        save_to_dir=validation_res_dir)
    
    n_validation_images = len(validation_generator.classes)
    n_batches = n_validation_images // tmp_batch_size

    res = model.predict_generator(
        validation_generator,
        n_batches,
        verbose=1)
    
    c_pred = np.round(res[:,0]).astype(int)
    c_correct = (c_pred == validation_generator.classes)
    pct_accuracy = np.sum(c_correct) / len(c_correct)
    print('Accuracy: {:.2f} %'.format(100.*pct_accuracy))
    print(len(validation_generator.classes))
    print(len(res[:,0]))

    txt = '\n'.join([
	'{:d} {:.0f} {:.5f}'.format(c, np.round(x), x)
	for c,x in zip(validation_generator.classes, res[:,0])])

    with open(os.path.join(validation_res_dir, 'predictions.txt'), 'w') as f:
        f.write(txt)
    
    print(res.shape)


def export_train_results():
    #model = create_network_v3()
    with open(os.path.join(weight_dir, 'first_try.json')) as f:
        model_json = f.read()
    model = kmodels.model_from_json(model_json)
    model.load_weights(os.path.join(weight_dir, 'first_try.h5'))
    #model = kmodels.load_model(os.path.join(weight_dir, 'first_try.h5'))
    print(model.summary())
    
    tmp_batch_size = 2

    # Compute the output for each validation
    train_datagen = kpimage.ImageDataGenerator(
        rescale=1./255.)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        color_mode='grayscale',
        target_size=(img_width, img_height),
        batch_size=tmp_batch_size,
        class_mode='binary',
	shuffle=False,
	save_to_dir=train_res_dir)
    
    n_train_images = len(train_generator.classes)
    n_batches = n_train_images // tmp_batch_size

    res = model.predict_generator(
	train_generator,
	n_batches,
	verbose=1)
   
    print(res.shape)
    c_pred = np.round(res[:,0]).astype(int)
    c_correct = (c_pred == train_generator.classes)
    pct_accuracy = np.sum(c_correct) / len(c_correct)
    print('Accuracy: {:.2f} %'.format(100.*pct_accuracy))
    print(len(train_generator.classes))
    print(len(res[:,0]))

    txt = '\n'.join([
	'{:d} {:.0f} {:.5f}'.format(c, np.round(x), x)
	for c,x in zip(train_generator.classes, res[:,0])])

    with open(os.path.join(train_res_dir, 'predictions.txt'), 'w') as f:
        f.write(txt)
    
    print(res.shape)


def main():
    if not len(data_dir):
        print('DATADIR not set!')
        return 0

    #train_full_network()
    export_validation_results(initial_epoch=84)
    #export_train_results()
    # save_bottleneck_features()
    # train_top_model()
    # fine_tune_top_model()
    
    #model = create_network_v3()
    #with open(os.path.join(weight_dir, 'first_try_old.json'), 'w') as f:
    #    f.write(model.to_json(indent=2))
    
    return 0


if __name__ == '__main__':
    main()
