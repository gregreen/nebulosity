#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import Grid
from itertools import cycle
import re


plot_colors = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b']

class_labels = [
    'nebulosity',
    'nebulosity_light',
    'normal',
    'sky_error']


def load_table(fname):
    with open(fname, 'r') as f:
        headers = f.readline().strip().split(',')
        dtype = [('epoch', 'u4')]
        dtype += [(s, 'f4') for s in headers[1:]]
        data = np.loadtxt(f, dtype=dtype, delimiter=',')

    return data

def get_n_classes(data):
    class_idx = []

    pattern = re.compile(r'[0-9]to[0-9]')
    for s,d in data.dtype.descr:
        if pattern.match(s):
            class_idx += [int(x) for x in s.split('to')]

    return max(class_idx) + 1


def get_a2x(data, class_origin, validation=False):
    if validation:
        prefix = 'val_'
    else:
        prefix = ''

    n_classes = get_n_classes(data)
    col_names = ['{}{}to{}'.format(prefix, class_origin, k)
                 for k in range(n_classes)]
    col_names[class_origin] = '{}comp{}'.format(prefix, class_origin)

    a2x = np.zeros((data.size, n_classes), dtype='f4')

    for k,col in enumerate(col_names):
        # if k == class_origin:
        #     continue
        a2x[:,k] = data[col][:]

    # a2x[:,class_origin] = 1. - np.sum(a2x, axis=1)
    a2x[:,:] /= np.sum(a2x, axis=1)[:,None]

    return a2x


def get_x2a(data, class_dest, validation=False):
    if validation:
        prefix = 'val_'
    else:
        prefix = ''

    n_classes = get_n_classes(data)
    col_names = ['{}{}to{}'.format(prefix, k, class_dest)
                 for k in range(n_classes)]

    x2a = np.zeros((data.size, n_classes), dtype='f4')

    for k,col in enumerate(col_names):
        if k == class_dest:
            continue
        x2a[:,k] = data[col][:]

    x2a[:,class_dest] = data['{}comp{}'.format(prefix, class_dest)]

    return x2a


def plot_stacked(data, class_idx,
                 validation=False, ax=None,
                 label=None, direction='a2x',
                 class_proportions=None):
    if direction == 'a2x':
        class_frac = get_a2x(data, class_idx, validation=validation)
    elif direction == 'x2a':
        class_frac = get_x2a(data, class_idx, validation=validation)
        if class_proportions is not None:
            for k,p in enumerate(class_proportions):
                class_frac[:,k] *= p
            class_frac[:,:] /= np.sum(class_frac, axis=1)[:,None]
    else:
        raise ValueError('<direction> must be either "a2x" or "x2a"')

    x = data['epoch'][:]
    y = np.cumsum(class_frac, axis=1)

    if ax is None:
        fig = plt.figure(figsize=(8,8), dpi=100)
        ax = fig.add_subplot(1,1,1)

    c = cycle(plot_colors)

    ax.fill_between(x, 0., y[:,0], alpha=0.5, facecolor=next(c))

    for k in range(1, y.shape[1]):
        ax.fill_between(x, y[:,k-1], y[:,k], alpha=0.5, facecolor=next(c))

    xlim = (min(x), max(x))

    if label is not None:
        if class_idx == 0:
            y_txt = 0.5 * y[-1,0]
        else:
            y_txt = 0.5 * (y[-1,class_idx] + y[-1,class_idx-1])

        w = max(x) - min(x)

        ax.text(xlim[1] - 0.025*w, y_txt, label,
                color=plot_colors[class_idx],
                ha='right',
                va='center',
                fontsize=14)

    ax.set_xlim(xlim)
    ax.set_ylim(0., 1.)


def main():
    fname = 'log/27th_try.log'
    class_proportions = [1420, 846, 1600, 503]

    class_props_normalized = np.array(class_proportions, dtype='f8')
    class_props_normalized /= np.sum(class_props_normalized)

    print('bits: {}'.format(-np.sum(class_props_normalized * np.log2(class_props_normalized))))
    return 0

    data = load_table(fname)

    # Plot what each class gets labeled as, and what classes each
    # label consists of

    titles = {
        'a2x': 'Images that are X are classified as ...',
        'x2a': 'Images classified as X are actually ...'}

    for direction in titles:
        fig = plt.figure(figsize=(10,10), dpi=100)

        grid = Grid(
            fig, 111,
            nrows_ncols=(2, 2),
            axes_pad=0.0)

        for k in range(4):
            plot_stacked(
                data, k,
                ax=grid[k],
                label=class_labels[k],
                direction=direction,
                class_proportions=class_proportions,
                validation=False)

            # grid[k].set_title(class_labels[k])

        grid[0].set_ylabel('fraction')
        grid[2].set_ylabel('fraction')
        grid[2].set_xlabel('epoch')
        grid[3].set_xlabel('epoch')

        fig.suptitle(titles[direction], fontsize=18)

        fig.savefig('plots/27th_try_' + direction + '.pdf',
                    bbox_inches='tight')

    # Plot loss
    fig = plt.figure(figsize=(8,5), dpi=100)
    ax = fig.add_subplot(1,1,1)

    ax.plot(data['epoch'], data['loss'],
            label='training loss', color=plot_colors[0],
            lw=2.0, ls='-', alpha=0.75)
    ax.plot(data['epoch'], data['val_loss'],
            label='validation loss', color=plot_colors[1],
            lw=2.0, ls='-', alpha=0.75)

    ax.legend()

    ax.set_xlabel('epoch')
    ax.set_ylabel('crossentropy (bits)')

    ax.set_xlim(np.min(data['epoch']), np.max(data['epoch']))

    fig.savefig('plots/27th_try_loss.pdf', bbox_inches='tight')

    plt.show()

    return 0


if __name__ == '__main__':
    main()
