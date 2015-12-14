import lasagne as nn
from lasagne.layers import NonlinearityLayer, BiasLayer, NINLayer, MaxPool2DLayer
from lasagne.nonlinearities import rectify
from lasagne_extensions.layers.batch_norm import BatchNormLayer
Conv2DLayer = nn.layers.Conv2DDNNLayer


def inception_module(l_in, num_1x1, num_2x1_proj, reduce_3x1, num_3x1, reduce_5x1, num_5x1, batch_norm=False, gain=1.0, bias=0.1):
    """
    Inception module for sequences
    :param l_in:
    :param num_1x1:
    :param num_2x1_proj:
    :param reduce_3x1:
    :param num_3x1:
    :param reduce_5x1:
    :param num_5x1:
    :param gain:
    :param bias:
    :return:
    """
    shape = l_in.get_output_shape()
    out_layers = []

    # 1x1
    if num_1x1 > 0:
        l_1x1 = NINLayer(l_in, num_units=num_1x1, W=nn.init.GlorotUniform(), b=None, nonlinearity=None, name='inception_1x1')
        l_1x1_bn = BatchNormalizeLayer(l_1x1, batch_norm)
        out_layers.append(l_1x1_bn)

    # 3x1
    if num_3x1 > 0:
        if reduce_3x1 > 0:
            l_reduce_3x1 = NINLayer(l_in, num_units=reduce_3x1, W=nn.init.GlorotUniform(), b=None, nonlinearity=None, name='inception_reduce_3x1')
            l_reduce_3x1_bn = BatchNormalizeLayer(l_reduce_3x1, batch_norm)
        else:
            l_reduce_3x1_bn = l_in
        l_3x1 = Conv2DLayer(l_reduce_3x1_bn, num_filters=num_3x1, filter_size=(3, 1), border_mode="same", W=nn.init.GlorotUniform(), b=None, nonlinearity=None, name='inception_3x1')
        l_3x1_bn = BatchNormalizeLayer(l_3x1, batch_norm)
        out_layers.append(l_3x1_bn)

    # 5x1
    if num_5x1 > 0:
        if reduce_5x1 > 0:
            l_reduce_5x1 = NINLayer(l_in, num_units=reduce_5x1, W=nn.init.GlorotUniform(), b=None, nonlinearity=None, name='inception_reduce_5x1')
            l_reduce_5x1_bn = BatchNormalizeLayer(l_reduce_5x1, batch_norm)
        else:
            l_reduce_5x1_bn = l_in
        l_5x1 = Conv2DLayer(l_reduce_5x1_bn, num_filters=num_5x1, filter_size=(3, 1), border_mode="same", W=nn.init.GlorotUniform(), b=None, nonlinearity=None, name='inception_3x1')
        l_5x1_bn = BatchNormalizeLayer(l_5x1, batch_norm)
        l_5x1 = Conv2DLayer(l_5x1_bn, num_filters=num_5x1, filter_size=(3, 1), border_mode="same", W=nn.init.GlorotUniform(), b=None, nonlinearity=None, name='inception_3x1')
        l_5x1_bn = BatchNormalizeLayer(l_5x1, batch_norm)
        out_layers.append(l_5x1_bn)

    if num_2x1_proj > 0:
        l_2x1_pool = MaxPool2DLayer(l_in, pool_size=(2, 1), name='inception_pool')
        l_2x1_proj = NINLayer(l_2x1_pool, num_units=num_2x1_proj, b=None, nonlinearity=None, name='inception_pool_proj')
        l_2x1_proj_bn = BatchNormalizeLayer(l_2x1_proj, batch_norm)
        out_layers.append(l_2x1_proj_bn)

    # stack
    l_out = nn.layers.concat(out_layers, name='Inception module')
    return l_out


def BatchNormalizeLayer(l_in, normalize=False, nonlinearity=rectify):
    if normalize:
        # l_prev = NormalizeLayer(l_prev, alpha='single_pass')
        # l_prev = ScaleAndShiftLayer(l_prev)
        # l_prev = NonlinearityLayer(l_prev, nonlinearity=nonlinearity)
        l_prev = BatchNormLayer(l_in, nonlinearity=nonlinearity, name='Batch norm')
    else:
        l_prev = NonlinearityLayer(l_in, nonlinearity=nonlinearity, name='Non-linearity')
        l_prev = BiasLayer(l_prev, name='Bias', b=nn.init.Constant(0.1))
    return l_prev
