import numpy as np
import theano.tensor as T

from lasagne import init
from lasagne.layers import Layer



class EmbeddingLayerNew(Layer):
    """
    lasagne.layers.EmbeddingLayer(incoming, input_size, output_size,
    W=lasagne.init.Normal(), **kwargs)

    A layer for word embeddings. The input should be an integer type
    Tensor variable.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.

    input_size: int
        The Number of different embeddings. The last embedding will have index
        input_size - 1.

    output_size : int
        The size of each embedding.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the embedding matrix.
        This should be a matrix with shape ``(input_size, output_size)``.
        See :func:`lasagne.utils.create_param` for more information.


    """
    def __init__(self, incoming, input_size, output_size,
                 W=init.Normal(), **kwargs):
        super(EmbeddingLayerNew, self).__init__(incoming, **kwargs)

        self.input_size = input_size
        self.output_size = output_size

        self.W = self.add_param(W, (input_size, output_size), name="W")

    def get_output_shape_for(self, input_shape):
        return input_shape + (self.output_size, )

    def get_output_for(self, input, **kwargs):
        input = input.astype('int32')
        return self.W[input]
