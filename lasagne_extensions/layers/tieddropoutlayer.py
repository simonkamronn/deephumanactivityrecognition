import lasagne as nn
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
_srng = RandomStreams()


class TiedDropoutLayer(nn.layers.Layer):
    """
    Dropout layer that broadcasts the mask across all axes beyond the first two.
    """
    def __init__(self, input_layer, p=0.5, rescale=True, **kwargs):
        super(TiedDropoutLayer, self).__init__(input_layer, **kwargs)
        self.p = p
        self.rescale = rescale

    def get_output_for(self, input, deterministic=False, *args, **kwargs):
        if deterministic or self.p == 0:
            return input
        else:
            retain_prob = 1 - self.p
            if self.rescale:
                input /= retain_prob

            mask = _srng.binomial(input.shape[:2], p=retain_prob,
                                  dtype=theano.config.floatX)
            axes = [0, 1] + (['x'] * (input.ndim - 2))
            mask = mask.dimshuffle(*axes)
            return input * mask

tied_dropout = TiedDropoutLayer