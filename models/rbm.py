__author__ = 'larsma'

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from .base import Model
import lasagne

class RBM(Model):
    def __init__(self,n_in=784,n_out=500,W=None,hbias=None,vbias=None,input=None,batch_size=100):
        super(RBM, self).__init__(n_in, [], n_out, batch_size, 'sigmoid')
        self.n_in = n_in
        self.n_out = n_out
        rng = np.random.RandomState(1234)
        theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.x = input
        if not input:
            self.x = T.matrix('x')
        self.l_hidden = lasagne.layers.DenseLayer(self.l_in, num_units=n_out, W=lasagne.init.Uniform() if W is None else W, b=lasagne.init.Constant(0.) if hbias is None else hbias, nonlinearity=self.transf)
        self.W = self.l_hidden.W
        self.l_visible = lasagne.layers.DenseLayer(self.l_hidden, num_units=n_in, W=self.W, b=lasagne.init.Constant(0.) if vbias is None else vbias, nonlinearity=self.transf)
        self.hbias = self.l_hidden.b
        self.vbias = self.l_visible.b
        self.theano_rng = theano_rng
        self.params = [self.W, self.hbias, self.vbias]

    def propup(self, vis):
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]


    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def get_cost_updates(self, lr=0.1, gibbs_steps=1):
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.x)
        chain_start = ph_sample
        ([pre_sigmoid_nvs,_,nv_samples,_,_,_],updates) = theano.scan(self.gibbs_hvh,outputs_info=[None, None, None, None, None, chain_start],n_steps=gibbs_steps)

        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.x)) - T.mean(
            self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            )
        # reconstruction cross-entropy is a good proxy for CD
        monitoring_cost = self.get_reconstruction_cost(updates,pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        cross_entropy = T.mean(
            T.sum(
                self.x * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.x) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy


    def build_model(self, *args):
        super(RBM, self).build_model_pretraining(*args)
        train_models = []
        test_models = []
        validate_models = []
        for rbm in self.rbm_layers:
            cost, updates = rbm.get_cost_updates(self.learning_rate, gibbs_steps=self.gibbs_steps)
            train_model = theano.function(
                inputs=[self.pretrain_index],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: self.train_x[self.pretrain_batch_slice]
                }
            )
            train_models.append(train_model)
            cost, updates = rbm.get_cost_updates(self.learning_rate, n_steps=self.gibbs_steps)
            test_model = theano.function(
                inputs=[self.pretrain_index],
                outputs=cost,
                givens={
                    self.x: self.test_x[self.pretrain_batch_slice]
                }
            )
            test_models.append(test_model)

            validate_model = None
            if not self.validation_x is None:
                validate_model = theano.function(
                        [self.pretrain_index],
                        outputs=cost,
                        givens={
                            self.x: self.validation_x[self.pretrain_batch_slice],
                        },
                    )
            validate_models.append(validate_model)

        return train_models, test_models, validate_models