#!/usr/bin/env python

from __future__ import print_function, division

import keras
import keras.backend as K
import keras.models as kmodels

import numpy as np
from PIL import Image
import os
import errno
import json
from glob import glob
import shutil
import random

from keras_diagram import ascii as model_to_ascii
from vis.utils import utils
from vis.visualization import visualize_activation, get_num_filters


def load_model(name, weight_dir):
    base_fname = os.path.join(weight_dir, name)

    with open(base_fname + '.json', 'r') as f:
        model_json = f.read()

    model = kmodels.model_from_json(model_json)
    model.load_weights(base_fname + '.h5')

    return model


def get_layer(model, layer_name):
    return model.get_layer(layer_name)


def get_layer_idx(model, layer_name):
    for k,layer in enumerate(model.layers):
        if layer.name == layer_name:
            return k
    return None


def tile_images(img_stack, nx, ny, dx, dy, fill=np.nan):
    h, w, n_images = img_stack.shape
    full_width = nx * (w + dx) - dx
    full_height = ny * (h + dy) - dy
    
    out_img = np.full(
        (full_height, full_width),
        fill,
        dtype=img_stack.dtype)
    
    i = 0
    for k in range(ny):
        y0 = k * (h + dy)
        for j in range(nx):
            x0 = j * (w + dx)
            out_img[y0:y0+h, x0:x0+w] = img_stack[:,:,i]
            i += 1
    
    return out_img


def tile_images_by_fname(img_fnames, w, h, nx, ny, dx, dy, fill=np.nan, max_size=128):
    img_stack = []

    for fn in img_fnames:
        img = Image.open(fn)
        img = expand_canvas(img, w, h)

        if max_size is not None:
            img = np.array(img)
            img = constrain_image_size(img, max_size)

        img_stack.append(np.array(img))
    
    img_stack = np.stack(img_stack, axis=-1)

    return tile_images(img_stack, nx, ny, dx, dy, fill=fill)


def weight_montage(model, layer_name, nx=None, ny=None, dx=1, dy=1, zoom=10):
    layer = get_layer(model, layer_name)
    weights = layer.get_weights()[0]
    # Collapse out axis 2
    #_, _, filter_depth, n_filters = weights.shape
    #idx2 = np.random.randint(filter_depth, size=n_filters)
    #idx3 = np.arange(n_filters)
    #weights = weights[...,idx2,idx3]
    weights.shape = (weights.shape[0], weights.shape[1], weights.shape[2]*weights.shape[3])

    weights /= np.abs(np.max(np.max(weights, axis=0), axis=0))[None, None, :]
    weights += 1.0
    weights *= 0.5
    
    # Find a good shape for the montage
    n_filters = weights.shape[2]
    if nx is None:
        nx, ny = find_good_shape(n_filters)

    img = tile_images(weights, nx, ny, dx, dy)
    img *= 255.
    img[~np.isfinite(img)] = 0.
    img = np.clip(img, 0., 255.).astype('u1')

    img = Image.fromarray(img, mode='L')
    
    if zoom is not None:
        new_shape = [int(s*zoom) for s in img.size]
        img = img.resize(new_shape, resample=Image.NEAREST)

    return img


def get_activation(model, layer_name, img):
    intermediate_model = kmodels.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_model.predict(img / 255.)
    return intermediate_output


def zoom_image(img, zoom):
    new_shape = [int(s*zoom) for s in img.size]
    img = img.resize(new_shape, resample=Image.NEAREST)
    return img


def resize_image_array(img, w, h, mode='L'):
    img = Image.fromarray(img, mode=mode)
    img = img.resize((w,h), resample=Image.LANCZOS)
    return np.array(img)


def constrain_image_size(img, max_scale, mode='L'):
    img_scale = max(img.shape)
    
    if img_scale <= max_scale:
        return img
    
    factor = max_scale / img_scale
    new_shape = [int(round(s*factor)) for s in img.shape]
    img = resize_image_array(img, new_shape[1], new_shape[0], mode=mode)
    
    return img


def find_good_shape(n_images):
    goodness = []

    for ny in range(1, int(np.floor(np.sqrt(n_images)))+1):
        nx = int(np.ceil(n_images / ny))
        goodness.append(-(nx/ny - 1.)**2. - 0.15*(n_images-nx*ny)**2.)

    ny = np.argmax(goodness) + 1
    nx = int(np.ceil(n_images / ny))

    return nx, ny


def expand_canvas(img, w, h):
    if img.size == (w, h):
        return img
    
    new_img = Image.new(img.mode, (w, h))
    new_img.paste(img, (0, 0, img.size[0], img.size[1]))
    
    return new_img


def get_activation_of_img_fname(model, layer_name, img_fname):
    # Load the image
    img = Image.open(img_fname)
    img = expand_canvas(img, 512, 512)
    img = np.array(img)

    # Fix the image shape (add channel and image axes)
    if K.image_data_format() == 'channels_first':
        img.shape = (1, 1, img.shape[0], img.shape[1])
    else:
        img.shape = (1, img.shape[0], img.shape[1], 1)
    
    # Calculate the activation of each filter
    activation = get_activation(model, layer_name, img)[0]
    
    return activation


def activation_montage(model, layer_name, img_fname, nx=None, ny=None, dx=2, dy=2, max_scale=1600):
    activation = get_activation_of_img_fname(model, layer_name, img_fname)

    # Find a good shape for the montage
    n_filters = activation.shape[2]
    if nx is None:
        nx, ny = find_good_shape(n_filters)
    
    # Generate montage of filter activations
    img = tile_images(activation, nx, ny, dx, dy)
    img[~np.isfinite(img)] = 0.
    img = (img.clip(0., 1.) * 255.).astype('u1')

    # Zoom and save the montage
    img = constrain_image_size(img, max_scale)
    img = Image.fromarray(img, mode='L')

    return img


def maxactivation_montage(model, layer_name, nx=None, ny=None, dx=4, dy=4, size=64):
    layer_idx = get_layer_idx(model, layer_name)
    n_filters = get_num_filters(model.layers[layer_idx])

    img_stack = np.empty((size,size,n_filters), dtype='u1')

    for k in range(n_filters):
        img = visualize_activation(model, layer_idx, filter_indices=k) 
        img.shape = img.shape[:2]
        img_stack[:,:,k] = resize_image_array(img, size, size)
    
    if nx is None:
        nx, ny = find_good_shape(n_filters)

    img = tile_images(img_stack, nx, ny, dx, dy, fill=0)
    img = Image.fromarray(img, mode='L')
    
    return img


def get_validation_results(model, validation_data_dir):
    _, w, h, _ = model.layers[0].input_shape
    validation_batch_size = 2

    # Compute the output for each validation
    validation_datagen = kpimage.ImageDataGenerator(
        rescale=1./255.)

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        color_mode='grayscale',
        target_size=(w, h),
        batch_size=validation_batch_size,
        class_mode='binary',
        shuffle=False,
        save_to_dir=validation_res_dir)
    
    n_validation_images = len(validation_generator.classes)
    n_batches = n_validation_images // validation_batch_size

    res = model.predict_generator(
        validation_generator,
        n_batches,
        verbose=1)
    
    c_pred = np.argmax(res[:,0])
    c_true = validation_generator.classes
    pct_accuracy = np.sum(c_pred == c_true) / len(c_pred)
    print('Accuracy: {:.2f} %'.format(100.*pct_accuracy))
    
    img_fnames = validation_generator.filenames

    return img_fnames, c_pred, c_true


def validation_montages(model, validation_data_dir):
    img_fnames, c_pred, c_true = get_validation_results(
        model, validation_data_dir)

    n_classes = model.layers[-1].output_shape[-1]
    
    for class_idx in range(n_classes):
        idx = (c_pred == c_true) & (c_pred == class_idx)
        

    # TODO: Finish this function



def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as err:
        if err.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise(err)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description='Make diagnostic plots for a neural network.',
        add_help=True)
    parser.add_argument('model', type=str, help='Name of neural network model.')
    parser.add_argument('layer', type=str, nargs='+', help='Layers to visualize.')
    parser.add_argument('-N', '--max-images', type=int,
        help='Max. # of images to visualize')
    parser.add_argument('--max-activation', action='store_true',
        help='Generate input images that maximize activation of different filters.')
    args = parser.parse_args()
    
    #for n in range(1, 25):
    #    nx, ny = find_good_shape(n)
    #    print('{: >2d} : {} x {}'.format(n, nx, ny))
    
    try:
        data_dir = os.environ['DATADIR']
    except KeyError as err:
        print('Environmental variable "DATADIR" must be set.')
        return 1
    
    # Find validation images
    validation_dir = os.path.join(data_dir, 'validation')
    img_classes = os.walk(validation_dir).next()[1]
    img_classes = sorted(img_classes)
    print('Found images classes: {}'.format(img_classes))

    validation_images = {}
    for key in img_classes:
        validation_images[key] = glob(os.path.join(validation_dir, key, '*.png'))
        random.shuffle(validation_images[key])
        print('Found {:d} images of class "{}"'.format(
              len(validation_images[key]), key))
    
    # Load the model
    model = load_model(args.model, os.path.join(data_dir, 'weights'))
    print('Loaded model "{}"'.format(args.model))
    print('')
    model.summary()
    print('')
    print(model_to_ascii(model))
    print('')
    
    # Set up output directory for plots
    plot_dir = os.path.join(data_dir, 'plots', 'diagnostics')
    mkdir_p(plot_dir)
    plot_dir = os.path.join(plot_dir, args.model)
    mkdir_p(plot_dir)
    
    # Visualize first flat layer
    flat_dir = os.path.join(plot_dir, 'flat_activations')
    mkdir_p(flat_dir)
    
    flat_activation = {}
    flat_layer_name = None
    flat_layer_size = None
    norm = []
    
    for layer in model.layers:
        if len(layer.output_shape) == 2:
            flat_layer_name = layer.name
            flat_layer_size = layer.output_shape[1]
            break
    
    print('')
    print('First flat layer: {}'.format(flat_layer_name))
    print('# of neurons: {}'.format(flat_layer_size))
    print('')
    
    for img_class in img_classes:
        print('First flat layer activations for class "{}" ...'.format(img_class))

        class_dir = os.path.join(plot_dir, img_class)
        mkdir_p(class_dir)
        
        n_images = len(validation_images[img_class])
        flat_activation[img_class] = np.empty((n_images, flat_layer_size), dtype='f8')

        for k,img_fname in enumerate(validation_images[img_class]):
            flat_activation[img_class][k,:] = get_activation_of_img_fname(
                model, flat_layer_name, img_fname)
        
        norm.append(np.percentile(flat_activation[img_class], 90.))

    norm = max(norm)

    for img_class in img_classes:
        img = (255. * np.clip(flat_activation[img_class]/norm, 0., 1.)).astype('u1')
        img = Image.fromarray(img, mode='L')
        img = zoom_image(img, 3)
        img_fname = 'flat_activation_{}.png'.format(img_class)
        img.save(os.path.join(flat_dir, img_fname))
    
    img = np.vstack([
        np.mean(flat_activation[c], axis=0)
        for c in img_classes])

    img /= np.percentile(img, 90.)
    img = (255. * np.clip(img, 0., 1.)).astype('u1')
    img = Image.fromarray(img, mode='L')
    img = zoom_image(img, 8)
    img_fname = 'flat_activation_average.png'
    img.save(os.path.join(flat_dir, img_fname))

    # Example images that maximally/minimally activate each neuron in 1st flat layer
    for k in range(flat_layer_size):
        neuron_activation = []
        img_fname = []

        for img_class in img_classes:
            neuron_activation.append(flat_activation[img_class][:,k])
            img_fname += validation_images[img_class]

        neuron_activation = np.hstack(neuron_activation)
        idx_sort = np.argsort(neuron_activation)

        img_fname = [img_fname[j] for j in idx_sort[-3:]] + [img_fname[j] for j in idx_sort[:3]]
        img = tile_images_by_fname(img_fname, 512, 512, 3, 2, 4, 4, max_size=128, fill=0)
        img = Image.fromarray(img, mode='L')

        img_fname = 'neuron_{:02d}.jpg'.format(k)
        print(img_fname)
        img.save(os.path.join(flat_dir, img_fname))


    # Generate montage of weights for each layer
    weights_dir = os.path.join(plot_dir, 'weights')
    mkdir_p(weights_dir)

    for layer_name in args.layer:
        if layer_name == '0':
            continue

        out_fname = os.path.join(weights_dir, layer_name+'.png')

        img = weight_montage(model, layer_name, dx=1, dy=1, zoom=5)
        img.save(out_fname)
    
    # Generate montage of max-activation input for each layer
    if args.max_activation:
        maxact_dir = os.path.join(plot_dir, 'max-activation')
        mkdir_p(maxact_dir)

        for layer_name in args.layer:
            if layer_name == '0':
                continue

            out_fname = os.path.join(maxact_dir, layer_name+'.jpg')

            img = maxactivation_montage(model, layer_name, dx=4, dy=4)
            img.save(out_fname)

    # Generate montage of activations for each image
    if args.max_images is not None:
        for key in img_classes:
            random.shuffle(validation_images[key])
            validation_images[key] = validation_images[key][:args.max_images]
    
    for img_class in img_classes:
        print('Activation montages for class "{}" ...'.format(img_class))

        class_dir = os.path.join(plot_dir, img_class)
        mkdir_p(class_dir)

        img_avg = {l: None for l in args.layer}

        for img_fname in validation_images[img_class]:
            img_name = os.path.splitext(os.path.basename(img_fname))[0]

            for layer_name in args.layer:
                out_fname = os.path.join(class_dir, img_name+'_'+layer_name+'.jpg')
                if layer_name == '0':
                    img = Image.open(img_fname)
                    img.save(out_fname)
                    #TODO: convert to JPEG
                    #shutil.copy2(img_fname, out_fname)
                else:
                    img = activation_montage(model, layer_name, img_fname, dx=2, dy=2)
                    img.save(out_fname)
                    
                    if img_avg[layer_name] is None:
                        img_avg[layer_name] = np.array(img).astype('f8') / 255.
                    else:
                        img_avg[layer_name] += np.array(img).astype('f8') / 255.
        
        n_images = len(validation_images[img_class])

        for layer_name in args.layer:
            if layer_name == '0':
                continue

            img = img_avg[layer_name]
            img /= float(n_images)
            img = (img.clip(0., 1.) * 255.).astype('u1')
            img = Image.fromarray(img, mode='L')

            img_fname = 'average_{}_{}.jpg'.format(img_class, layer_name)
            img.save(os.path.join(class_dir, img_fname))

    return 0


if __name__ == '__main__':
    main()
