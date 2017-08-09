#!/usr/bin/env python

from __future__ import print_function, division

import keras
import keras.models as kmodels

import numpy as np

from equalize import histogram_equalization

#import astropy.io.fits as fits


def load_model(fname_base):
    with open(fname_base + '.json', 'r') as f:
        model_json = f.read()

    model = kmodels.model_from_json(model_json)
    model.load_weights(fname_base + '.h5')

    return model


def subimages(img, shape):
    j = np.arange(0, img.shape[0]+shape[0]-1, shape[0], dtype=int)
    k = np.arange(0, img.shape[1]+shape[1]-1, shape[1], dtype=int)

    for j0,j1 in zip(j[:-1], j[1:]):
        for k0,k1 in zip(k[:-1], k[1:]):
            yield j0, k0, img[j0:j1, k0:k1]

def pad_to(img, shape):
    if img.shape != shape:
        ret = np.zeros(shape, dtype=img.dtype)
        ret[:img.shape[0], :img.shape[1]] = img[:,:]
        return ret
        #padding = [s1-s0 for s0,s1 in zip(img.shape, shape)]
        #print(img.shape)
        #print(shape)
        #print(padding)
        #print('')
        #return np.pad(img, padding, 'constant', constant_values=0.)
    else:
        return img



def gen_mask(model, img):
    _, h, w, _ = model.layers[0].input_shape
    
    mask = np.empty(img.shape, dtype='u1')

    for j0,k0,subimg in subimages(img, (h, w)):
        subimg, _ = histogram_equalization(subimg.astype('f8'))
        subimg /= 255.
        subimg = pad_to(subimg, (h,w))
        subimg.shape = (1, subimg.shape[0], subimg.shape[1], 1)
        pred = model.predict(subimg, batch_size=1)[0]
        print('Confidence: {:.0f}% nebulosity, {:.0f}% normal, {:.0f}% sky_error'.format(
            100.*pred[0], 100.*pred[1], 100.*pred[2]))
        mask[j0:j0+h, k0:k0+w] = np.argmax(pred)

    return mask


def main():
    from PIL import Image
    
    model = load_model('toy_data/21st_try')
    
    img = Image.open('toy_data/c4d_160326_090015_ooi_z_v1.fits.fz.45.31.png')
    img = np.array(img)
    
    mask = gen_mask(model, img)
    mask = Image.fromarray((255.*mask/2.).astype('u1'), mode='L')
    mask.save('toy_data/test_image_mask.png')

    return 0


if __name__ == '__main__':
    main()
