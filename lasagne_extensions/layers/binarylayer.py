import theano.tensor as T
import lasagne


class BinaryLayer(lasagne.layers.MergeLayer):
    def __init__(self, incoming, min=0, max=1, **kwargs):
        super(BinaryLayer, self).__init__([incoming], **kwargs)
        self.min = min
        self.max = max

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        return T.iround(T.clip(input.pop(0), self.min, self.max), mode="half_away_from_zero")