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

from metrics import categorical_completeness, categorical_purity, categorical_cross

import os
import json
from glob import glob


# Training/validation data properties
data_dir = os.environ['DATADIR']
project_dir = os.path.expanduser('~/projects/nebulosity')

train_data_dir = os.path.join(data_dir, 'train/')
validation_data_dir = os.path.join(data_dir, 'validation/')
dataflow_dir = os.path.join(data_dir, 'dataflow/')
weight_dir = os.path.join(data_dir, 'weights/')
log_dir = os.path.join(data_dir, 'log/')
validation_res_dir = os.path.join(data_dir, 'validation_res/')
train_res_dir = os.path.join(data_dir, 'train_res/')
checkpoint_dir = os.path.join(data_dir, 'checkpoints/')


#train_samples = 2 * 1200 + 375
#validation_samples = 2 * 307 + 61


# Image properties
img_width, img_height = (512, 512) #(512, 512)
if kbackend.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)


# Training settings
epochs = 150
batch_size = 32
output_name = '27th_try'
class_names = ['nebulosity', 'nebulosity_light', 'normal', 'sky_error']


def get_class_info():
    #class_names = [
    #    s.rstrip('/')
    #    for s in glob(os.path.join(data_dir, 'train', '*/'))]
    class_train_samples = {
        s: len(glob(os.path.join(train_data_dir, s, '*.png')))
        for s in class_names}
    class_validation_samples = {
        s: len(glob(os.path.join(validation_data_dir, s, '*.png')))
        for s in class_names}

    print(class_train_samples)
    print(class_validation_samples)
    
    n_samples_max = max(class_train_samples.values())

    class_weight = {
        k: n_samples_max/v
        for k,v in class_train_samples.items()}
    
    train_samples = sum(class_train_samples.values())
    val_samples = sum(class_validation_samples.values())
    return class_weight, train_samples, val_samples


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
    # Get info about classes
    class_weight = get_class_info()[0]
    n_classes = len(class_weight)

    # Define the neural network architecture
    model = kmodels.Sequential()
    
    penalty = 0.001

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
        32, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        32, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(klayers.Conv2D(
        48, (3, 3),
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
        9,
        activation='relu',
        kernel_regularizer=kregularizers.l2(penalty),
        bias_regularizer=kregularizers.l2(penalty)
    ))
    model.add(klayers.Dropout(0.1))
    model.add(klayers.Dense(
        n_classes,
        activation='softmax',
        kernel_regularizer=kregularizers.l2(penalty),
        bias_regularizer=kregularizers.l2(penalty)
    ))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=koptimizers.Adadelta(lr=0.7),#'adadelta',
        metrics=[
            'accuracy',
            categorical_purity(0),
            categorical_completeness(0),
            categorical_cross(0,2),
            categorical_cross(2,0),
            categorical_cross(1,2),
            categorical_cross(2,1)
        ])

    return model


def create_network_v9():
    # Get info about classes
    class_weight = get_class_info()[0]
    n_classes = len(class_weight)

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
        32, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        32, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(klayers.Conv2D(
        32, (3, 3),
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
        9,
        activation='relu',
        kernel_regularizer=kregularizers.l2(penalty),
        bias_regularizer=kregularizers.l2(penalty)
    ))
    model.add(klayers.Dropout(0.1))
    model.add(klayers.Dense(
        n_classes,
        activation='softmax',
        kernel_regularizer=kregularizers.l2(penalty),
        bias_regularizer=kregularizers.l2(penalty)
    ))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=koptimizers.Adadelta(lr=0.07),#'adadelta',
        metrics=[
            'accuracy',
            categorical_purity(0),
            categorical_completeness(0),
            categorical_purity(2),
            categorical_completeness(2),
            categorical_cross(0,2),
            categorical_cross(2,0),
            categorical_cross(1,2),
            categorical_cross(2,1)
        ])

    return model


def create_network_v10():
    # Get info about classes
    class_weight = get_class_info()[0]
    n_classes = len(class_weight)

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
    model.add(klayers.Conv2D(
        24, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.Conv2D(
        24, (1, 1),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        32, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.Conv2D(
        32, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.Conv2D(
        32, (1, 1),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))

    model.add(klayers.Conv2D(
        32, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.Conv2D(
        32, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.Conv2D(
        32, (1, 1),
        activation='relu',
        padding='same',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(klayers.Conv2D(
        32, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.Conv2D(
        32, (3, 3),
        activation='relu',
        kernel_regularizer=kregularizers.l2(penalty)))
    model.add(klayers.Conv2D(
        32, (1, 1),
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
        3*n_classes,
        activation='relu',
        kernel_regularizer=kregularizers.l2(penalty),
        bias_regularizer=kregularizers.l2(penalty)
    ))
    model.add(klayers.Dropout(0.1))
    model.add(klayers.Dense(
        n_classes,
        activation='softmax',
        kernel_regularizer=kregularizers.l2(penalty),
        bias_regularizer=kregularizers.l2(penalty)
    ))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=koptimizers.Adadelta(lr=0.70),#'adadelta',
        metrics=[
            'accuracy',
            categorical_purity(0),
            categorical_completeness(0),
            categorical_purity(1),
            categorical_completeness(1),
            categorical_purity(2),
            categorical_completeness(2),
            categorical_purity(3),
            categorical_completeness(3),
            categorical_cross(0,1),
            categorical_cross(0,2),
            categorical_cross(0,3),
            categorical_cross(1,0),
            categorical_cross(1,2),
            categorical_cross(1,3),
            categorical_cross(2,0),
            categorical_cross(2,1),
            categorical_cross(2,3),
            categorical_cross(3,0),
            categorical_cross(3,1),
            categorical_cross(3,2),
        ])

    return model


def train_full_network(initial_epoch=0):
    """
    Train a full network from scratch.
    """

    class_weight, train_samples, val_samples = get_class_info()
    class_weight = [class_weight[k] for k in sorted(class_weight.keys())]
    print('Using class weights: {}'.format(class_weight))

    model = create_network_v10()
    model.summary()
    print('')
    print(model_to_ascii(model))
    print('')

    with open(os.path.join(log_dir, output_name + '_model_diagram.log'), 'w') as f:
        f.write(model_to_ascii(model))
    
    with open(os.path.join(weight_dir, output_name + '.json'), 'w') as f:
        f.write(model.to_json(indent=2))
    
    if initial_epoch != 0:
        print('Loading stored weights ...')
        model_fn = os.path.join(weight_dir, output_name+'.h5')
        model.load_weights(model_fn)
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
        classes=class_names,
        #save_to_dir=dataflow_dir,
        class_mode='categorical')

    validation_datagen = kpimage.ImageDataGenerator(
        rescale=1./255.)

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        color_mode='grayscale',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        classes=class_names,
        class_mode='categorical')
    
    checkpoint_fname = os.path.join(
        checkpoint_dir,
        output_name + '.epoch{epoch:03d}-loss{val_loss:0.3f}.h5')
    
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs+initial_epoch,
        validation_data=validation_generator,
        validation_steps=val_samples // batch_size,
        class_weight=class_weight,
        initial_epoch=initial_epoch,
        callbacks=[
            kcallbacks.CSVLogger(
                os.path.join(log_dir, output_name+'_initepoch{}'.format(initial_epoch)+'.log')),
            kcallbacks.EarlyStopping(monitor='val_loss', min_delta=0.005, patience=15),
            kcallbacks.ReduceLROnPlateau(monitor='val_loss', epsilon=0.01, patience=10, verbose=1),
            kcallbacks.ModelCheckpoint(checkpoint_fname)
        ])
    
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
        classes=class_names,
        class_mode='categorical',
        shuffle=False,
        save_to_dir=validation_res_dir)
    
    n_validation_images = len(validation_generator.classes)
    n_batches = n_validation_images // tmp_batch_size

    res = model.predict_generator(
        validation_generator,
        n_batches,
        verbose=1)
    
    c_pred = np.argmax(res[:,0])
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

    train_full_network(initial_epoch=77)
    #export_validation_results()
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
