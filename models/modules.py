import lasagne
from lasagne.layers import NonlinearityLayer, BiasLayer, NINLayer, MaxPool2DLayer, Conv2DLayer, ConcatLayer, \
    ElemwiseSumLayer
from lasagne.nonlinearities import rectify
from lasagne_extensions.layers.batch_norm import BatchNormLayer
from lasagne_extensions.layers import TiedDropoutLayer


def inception_module(l_in, num_1x1, num_3x1_proj, reduce_3x1, num_3x1, reduce_5x1, num_5x1,
                     batch_norm=False, gain=1.0, bias=0.1, nonlinearity=rectify):
    """
    Inception module for sequences
    :param l_in:
    :param num_1x1:
    :param num_3x1_proj:
    :param reduce_3x1:
    :param num_3x1:
    :param reduce_5x1:
    :param num_5x1:
    :param gain:
    :param bias:
    :return:
    """
    out_layers = []

    # 1x1
    if num_1x1 > 0:
        l_1x1 = NINLayer(l_in, num_units=num_1x1, W=lasagne.init.GlorotUniform(),
                         b=None, nonlinearity=None, name='inception_1x1')
        l_1x1_bn = BatchNormalizeLayer(l_1x1, batch_norm, nonlinearity)
        out_layers.append(l_1x1_bn)

    # 3x1
    if num_3x1 > 0:
        if reduce_3x1 > 0:
            l_reduce_3x1 = NINLayer(l_in, num_units=reduce_3x1, W=lasagne.init.GlorotUniform(),
                                    b=None, nonlinearity=None, name='inception_reduce_3x1')
            l_reduce_3x1_bn = BatchNormalizeLayer(l_reduce_3x1, batch_norm, nonlinearity)
        else:
            l_reduce_3x1_bn = l_in
        l_3x1 = Conv2DLayer(l_reduce_3x1_bn, num_filters=num_3x1, filter_size=(3, 1), pad="same",
                            W=lasagne.init.GlorotUniform(), b=None, nonlinearity=None, name='inception_3x1')
        l_3x1_bn = BatchNormalizeLayer(l_3x1, batch_norm, nonlinearity)
        out_layers.append(l_3x1_bn)

    # 5x1
    if num_5x1 > 0:
        if reduce_5x1 > 0:
            l_reduce_5x1 = NINLayer(l_in, num_units=reduce_5x1, W=lasagne.init.GlorotUniform(),
                                    b=None, nonlinearity=None, name='inception_reduce_5x1')
            l_reduce_5x1_bn = BatchNormalizeLayer(l_reduce_5x1, batch_norm, nonlinearity)
        else:
            l_reduce_5x1_bn = l_in

        l_5x1 = Conv2DLayer(l_reduce_5x1_bn, num_filters=num_5x1, filter_size=(3, 1), pad="same",
                            W=lasagne.init.GlorotUniform(), b=None, nonlinearity=None, name='inception_5x1/1')
        l_5x1_bn = BatchNormalizeLayer(l_5x1, batch_norm, nonlinearity)

        l_5x1 = Conv2DLayer(l_5x1_bn, num_filters=num_5x1, filter_size=(3, 1), pad="same",
                            W=lasagne.init.GlorotUniform(), b=None, nonlinearity=None, name='inception_5x1/2')
        l_5x1_bn = BatchNormalizeLayer(l_5x1, batch_norm, nonlinearity)
        out_layers.append(l_5x1_bn)

    if num_3x1_proj > 0:
        l_3x1_pool = MaxPool2DLayer(l_in, pool_size=(3, 1), stride=(1, 1), pad=(1, 0), name='inception_pool')
        l_3x1_proj = NINLayer(l_3x1_pool, num_units=num_3x1_proj, b=None, nonlinearity=None, name='inception_pool_proj')
        l_3x1_proj_bn = BatchNormalizeLayer(l_3x1_proj, batch_norm, nonlinearity)
        out_layers.append(l_3x1_proj_bn)

    # stack
    l_out = ConcatLayer(out_layers, axis=1, name='Inception module')
    return l_out


def ResidualModule(input_layer, num_filters=64, nonlinearity=rectify, normalize=False, stride=(1, 1), conv_dropout=0.0):
    input_conv = Conv2DLayer(incoming=input_layer,
                             num_filters=num_filters,
                             filter_size=(3, 1),
                             stride=stride,
                             pad='same',
                             W=lasagne.init.GlorotUniform(),
                             nonlinearity=None,
                             b=None,
                             name='Residual module layer 1')
    l_prev = BatchNormalizeLayer(input_conv, normalize=normalize, nonlinearity=nonlinearity)
    l_prev = TiedDropoutLayer(l_prev, p=conv_dropout, name='Tied Dropout')

    l_prev = Conv2DLayer(incoming=l_prev,
                         num_filters=num_filters,
                         filter_size=(3, 1),
                         stride=(1, 1),
                         pad='same',
                         W=lasagne.init.GlorotUniform(),
                         nonlinearity=None,
                         b=None,
                         name='Residual module layer 2')
    if normalize:
        # Batch normalization is done "immediately after" convolutions
        l_prev = BatchNormLayer(l_prev, name='Batch norm')

    # Using 1x1 convolutions for shortcut projections. NiNLayer could be used as well
    # but doesn't' support strides
    l_skip = Conv2DLayer(input_layer, num_filters=num_filters, filter_size=(1, 1), stride=stride,
                         nonlinearity=None, b=None, name='Shortcut')
    l_prev = ElemwiseSumLayer((l_prev, l_skip), name='Elementwise sum')

    # Add nonlinearity after summation
    l_prev = NonlinearityLayer(l_prev, nonlinearity=nonlinearity, name='Non-linearity')
    if not normalize:
        l_prev = BiasLayer(l_prev, name='Bias')

    l_prev = TiedDropoutLayer(l_prev, p=conv_dropout, name='Tied Dropout')
    return l_prev


def RecurrentConvLayer(input_layer, t=3, num_filters=64, filter_size=7, nonlinearity=rectify,
                       normalize=False, rcl_dropout=0.0):
    input_conv = Conv2DLayer(incoming=input_layer,
                             num_filters=num_filters,
                             filter_size=(1, 1),
                             stride=(1, 1),
                             pad='same',
                             W=lasagne.init.GlorotNormal(),
                             nonlinearity=None,
                             b=None,
                             name='RCL Conv2D input')
    l_prev = BatchNormalizeLayer(input_conv, normalize=normalize, nonlinearity=nonlinearity)
    l_prev = TiedDropoutLayer(l_prev, p=rcl_dropout, name='Recurrent conv dropout')

    for _ in range(t):
        l_prev = Conv2DLayer(incoming=l_prev,
                             num_filters=num_filters,
                             filter_size=(filter_size, 1),
                             stride=(1, 1),
                             pad='same',
                             W=lasagne.init.GlorotNormal(),
                             nonlinearity=None,
                             b=None,
                             name='Conv2D t=%d' % _)
        l_prev = ElemwiseSumLayer((input_conv, l_prev), coeffs=1, name='Sum')
        l_prev = BatchNormalizeLayer(l_prev, normalize=normalize, nonlinearity=nonlinearity)
        l_prev = TiedDropoutLayer(l_prev, p=rcl_dropout, name='Recurrent conv dropout')
    return l_prev


def BatchNormalizeLayer(l_prev, normalize=False, nonlinearity=rectify):
    """
    Batch normalise or add non-linearity and bias
    :param l_prev: input layer
    :param normalize: True or False
    :param nonlinearity: non-linearity to apply
    :return:
    """
    if normalize:
        # l_prev = NormalizeLayer(l_prev, alpha='single_pass')
        # l_prev = ScaleAndShiftLayer(l_prev)
        # l_prev = NonlinearityLayer(l_prev, nonlinearity=nonlinearity)
        l_prev = BatchNormLayer(l_prev, name='Batch norm')
        l_prev = NonlinearityLayer(l_prev, nonlinearity=nonlinearity, name='Non-linearity')
    else:
        l_prev = NonlinearityLayer(l_prev, nonlinearity=nonlinearity, name='Non-linearity')
        l_prev = BiasLayer(l_prev, name='Bias')
    return l_prev
