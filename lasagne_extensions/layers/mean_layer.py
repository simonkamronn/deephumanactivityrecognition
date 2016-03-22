__author__ = 'larsma'

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers.base import Layer
import math
from theano.tensor.shared_randomstreams import RandomStreams


class MeanLayer(lasagne.layers.MergeLayer):
    def __init__(self, incommings, axis, **kwargs):
        super(MeanLayer, self).__init__([incommings], **kwargs)

        self.axis = axis

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:self.axis]+input_shapes[0][self.axis+1:]

    def get_output_for(self, input, deterministic=False, **kwargs):
        return input[0].mean(axis=self.axis)