#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np


def histogram_equalization(img, n_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    img_histogram, bins = np.histogram(img.flatten(), n_bins, normed=True)
    cdf = img_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    img_equalized = np.interp(img.flatten(), bins[:-1], cdf)

    return img_equalized.reshape(img.shape), cdf


def main():
    from PIL import Image
    img = Image.open('data/sample_catastrophic.jpg')
    img = np.array(img)
    img, _ = histogram_equalization(img)
    img = Image.fromarray(img.astype('u1'), mode='L')
    img.save('data/sample_catastrophic.png')


if __name__ == '__main__':
    main()
