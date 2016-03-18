import lasagne
import theano.tensor as T


class ConstrainLayer(lasagne.layers.MergeLayer):
    def __init__(self, incoming, scale=1, min=-1e2, max=1e2, **kwargs):
        super(ConstrainLayer, self).__init__([incoming, scale, min, max], **kwargs)
        self.scale = scale
        self.min = min
        self.max = max

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        return T.clip(input * self.scale, self.min, self.max)