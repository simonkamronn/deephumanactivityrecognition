import theano
import lasagne


class OneHotLayer(lasagne.layers.Layer):
    def __init__(self, incoming, nb_class, **kwargs):
        super(OneHotLayer, self).__init__(incoming, **kwargs)
        self.nb_class = nb_class

    def get_output_for(self, incoming, **kwargs):
        return theano.tensor.extra_ops.to_one_hot(incoming, self.nb_class)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.nb_class)
