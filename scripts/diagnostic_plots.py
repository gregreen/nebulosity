#!/usr/bin/env python

from __future__ import print_function, division

import keras
import keras.backend as K
import keras.models as kmodels

import numpy as np
from PIL import Image
import os
import json


data_dir = os.environ['DATADIR']
weight_dir = os.path.join(data_dir, 'weights')
plot_dir = os.path.join(data_dir, 'plots')


def load_model(name):
    base_fname = os.path.join(weight_dir, name)

    with open(base_fname + '.json', 'r') as f:
        model_json = f.read()

    model = kmodels.model_from_json(model_json)
    model.load_weights(base_fname + '.h5')

    return model


def get_layer(model, layer_name):
    return model.get_layer(layer_name)
    #for layer in model.layers:
    #    if layer.name == layer_name:
    #        return layer
    #return None


def tile_images(img_stack, nx, ny, dx, dy):
    w, h, n_images = img_stack.shape
    full_width = nx * (w + dx) - dy
    full_height = ny * (h + dy) - dy
    print(w, h, n_images)
    
    out_img = np.full((full_width, full_height), np.nan)
    
    i = 0
    for j in range(nx):
        x0 = j * (w + dx)
        for k in range(ny):
            y0 = k * (h + dy)
            out_img[x0:x0+w, y0:y0+h] = img_stack[:,:,i]
            i += 1
    
    return out_img


def weight_montage(model, layer_name, nx, ny, dx=1, dy=1, zoom=10):
    layer = get_layer(model, layer_name)
    weights = layer.get_weights()[0]
    weights.shape = (weights.shape[0], weights.shape[1], weights.shape[3])
    print(weights)
    print(weights.shape)

    weights /= np.abs(np.max(np.max(weights, axis=0), axis=0))[None, None, :]
    weights += 1.0
    weights *= 0.5

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


def activation_montage(model, layer_name, img_fname, nx, ny, dx=1, dy=1, zoom=2):
    img = Image.open(img_fname)
    img = np.array(img)

    if K.image_data_format() == 'channels_first':
        img.shape = (1, 1, img.shape[0], img.shape[1])
    else:
        img.shape = (1, img.shape[0], img.shape[1], 1)
    
    activation = get_activation(model, layer_name, img)[0]
    print(np.min(activation), np.max(activation))

    print(activation.shape)
    
    img = tile_images(activation, nx, ny, dx, dy)
    img[~np.isfinite(img)] = 0.
    img = (img.clip(0., 1.) * 255.).astype('u1')

    img = Image.fromarray(img, mode='L')
    if zoom is not None:
        new_shape = [int(s*zoom) for s in img.size]
        img = img.resize(new_shape, resample=Image.NEAREST)

    return img


def main():
    model = load_model('sixth_try')
    model.summary()
    
    #img = weight_montage(model, 'conv2d_1', 4, 8)
    #img.save('conv2d_1.png')
    
    for img_name in ['sample_nebulosity', 'sample_faint_nebulosity', 'sample_normal', 'sample_crazy']:
        layer_name = 'conv2d_4'
        img_fname = os.path.join(data_dir, 'data', img_name+'.png')
        img = activation_montage(model, layer_name, img_fname, 7, 7, dx=2, dy=2)
        img.save(os.path.join(
            plot_dir,
            '{}_{}_activation.png'.format(layer_name, img_name)))

    return 0


if __name__ == '__main__':
    main()
