#!/usr/bin/env python

from __future__ import print_function, division

import keras
import keras.backend as K


def count_true(x):
    """
    Count the number of true elements in a boolean tensor.

    Uses the Keras backend, and returns a float.
    """
    return K.cast(
        K.sum(K.cast(x, dtype='int64')),
        dtype=K.floatx())


def categorical_purity(class_idx):
    """
    Returns a Keras metric measuring the purity of a given class.

    This function works with one-hot-encoded output.
    
    Purity is defined as the fraction of objects predicted to be
    of the class that actually are of the class.
    """
    def purity(y_true, y_pred):
        true_class = K.equal(
            K.argmax(y_true, axis=-1),
            class_idx)
        pred_class = K.equal(
            K.argmax(y_pred, axis=-1),
            class_idx)
        numerator = count_true(pred_class & true_class)
        denominator = count_true(pred_class)
        return numerator / (denominator + K.epsilon())

    purity.__name__ = 'pure{:d}'.format(class_idx)
    return purity


def categorical_completeness(class_idx):
    """
    Returns a Keras metric measuring the completeness of a given class.

    This function works with one-hot-encoded output.
    
    Completeness is defined as the fraction of objects that are of a
    given class that are predicted to be part of the class.
    """
    def completeness(y_true, y_pred):
        true_class = K.equal(
            K.argmax(y_true, axis=-1),
            class_idx)
        pred_class = K.equal(
            K.argmax(y_pred, axis=-1),
            class_idx)
        numerator = count_true(pred_class & true_class)
        denominator = count_true(true_class)
        return numerator / (denominator + K.epsilon())
    
    completeness.__name__ = 'comp{:d}'.format(class_idx)
    return completeness


def categorical_cross(class_from, class_to):
    """
    Returns a Keras metric measuring the fraction of objects of one
    class that get categorized as another class.

    This function works with one-hot-encoded output.
    """
    def cross(y_true, y_pred):
        true_from = K.equal(
            K.argmax(y_true, axis=-1),
            class_from)
        pred_to = K.equal(
            K.argmax(y_pred, axis=-1),
            class_to)
        numerator = count_true(true_from & pred_to)
        denominator = count_true(true_from)
        return numerator / (denominator + K.epsilon())

    cross.__name__ = '{:d}to{:d}'.format(class_from, class_to)
    return cross
