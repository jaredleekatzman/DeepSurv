import lasagne
import numpy
import time

import theano
import theano.tensor as T

from lifelines.utils import concordance_index

from lasagne.regularization import regularize_layer_params, l1, l2

class DeepSurv:
    def __init__(self, n_in,
    learning_rate, lr_decay, L2_reg, L1_reg = 0.00,
    momentum = 0.9,
    hidden_layers_sizes = None,
    activation = lasagne.nonlinearities.rectify,
    batch_norm = True,
    dropout = None,
    standardize = False,
    ):

        self.X = T.fmatrix('x')  # patients covariates
        self.E = T.ivector('e') # the observations vector

        # self.offset = T.mean(self.X, axis = 0)
        # self.scale = T.std(self.X, axis = 0)

        # self.offset = T.fvector('offset')
        # self.scale = T.fvector('scale')

        self.offset = theano.shared(numpy.zeros(shape = n_in, dtype=numpy.float32))
        self.scale = theano.shared(numpy.ones(shape = n_in, dtype=numpy.float32))

        network = lasagne.layers.InputLayer(shape=(None,n_in),
        input_var = self.X)

        if standardize:
            network = lasagne.layers.standardize(network,self.offset,
                                                self.scale,
                                                shared_axes = 0)
        self.standardize = standardize

        # Construct Neural Network
        for n_layer in (hidden_layers_sizes or []):
            if activation == lasagne.nonlinearities.rectify:
                W_init = lasagne.init.GlorotUniform()
            else:
                W_init = lasagne.init.GlorotUniform()


            network = lasagne.layers.DenseLayer(
                network, num_units = n_layer,
                nonlinearity = activation,
                W = W_init
            )

            if batch_norm:
                network = lasagne.layers.batch_norm(network)

            if not dropout is None:
                network = lasagne.layers.DropoutLayer(network, p = dropout)

        # Combine Linear to output Log Hazard Ratio - same as Faraggi
        network = lasagne.layers.DenseLayer(
            network, num_units = 1,
            nonlinearity = lasagne.nonlinearities.linear,
            W = lasagne.init.GlorotUniform()
        )

        self.network = network
        self.params = lasagne.layers.get_all_params(self.network,
                                                    trainable = True)
        self.hidden_layers = lasagne.layers.get_all_layers(self.network)[1:]


        self.risk_score = T.exp(self.log_hr(deterministic = True))

        # Set Hyper-parameters:
        self.n_in = n_in
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.L2_reg = L2_reg
        self.L1_reg = L1_reg
        self.momentum = momentum

    def load_model(self, params):
        lasagne.layers.set_all_param_values(self.network, params, trainable=True)

    def log_hr(self,deterministic = False):
        return lasagne.layers.get_output(self.network,
                                        deterministic = deterministic)

    def get_loss_updates(self,
    L1_reg = 0.0, L2_reg = 0.001,
    update_fn = lasagne.updates.nesterov_momentum,
    max_norm = None, deterministic = False,
    **kwargs):

        loss = (
            self.negative_log_likelihood(self.E, deterministic)
            + regularize_layer_params(self.network,l1) * L1_reg
            + regularize_layer_params(self.network, l2) * L2_reg
            # + L2_reg * self.L2_sqr()
        )

        if max_norm:
            grads = T.grad(loss,self.params)
            scaled_grads = lasagne.updates.total_norm_constraint(grads, max_norm)
            updates = update_fn(
                grads, self.params, **kwargs
            )
            return loss, updates

        updates = update_fn(
                loss, self.params, **kwargs
            )

        return loss, updates

    def negative_log_likelihood(self, E, deterministic = False):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \sum_{i \in D}[F(x_i,\theta) - log(\sum_{j \in R_i} e^F(x_j,\theta))]
                - \lambda P(\theta)

        where:
            D is the set of observed events
            R_i is the set of examples that are still alive at time of death t_j
            F(x,\theta) = log hazard rate
            P(\theta) = regularization equation
            \lamba = regularization coefficient


        :type E: theano.tensor.TensorType
        :param E: corresponds to a vecotr that gives the censor variable for
                  each example

        Note: we assume that the training examples are sorted in order of E
                (this is because we use the theano.cumsum function to get the
                overall risk associated with an individual example)
            Additionally, we assume that no patients die on the same day
                (otherwise likelihood would be different)
        """
        log_hr = self.log_hr(deterministic)
        hazard_ratio = T.exp(log_hr)
        log_risk = T.log(T.extra_ops.cumsum(hazard_ratio))
        uncensored_likelihood = log_hr.T - log_risk
        censored_likelihood = uncensored_likelihood * E
        return -T.sum(censored_likelihood)

    def get_train_valid_fn(self,
    L1_reg, L2_reg, learning_rate,
    **kwargs):
        loss, updates = self.get_loss_updates(
            L1_reg, L2_reg, deterministic = False,
            learning_rate=learning_rate, **kwargs
        )
        train_fn = theano.function(
            inputs = [self.X, self.E],
            outputs = loss,
            updates = updates,
            name = 'train'
        )

        valid_loss, _ = self.get_loss_updates(
            L1_reg, L2_reg, deterministic = True,
            learning_rate=learning_rate, **kwargs
        )

        valid_fn = theano.function(
            inputs = [self.X, self.E],
            outputs = valid_loss,
            name = 'valid'
        )
        return train_fn, valid_fn

    def get_concordance_index(self, data, true_survival, event_observed):
        """
        Taken from the lifelines.utils package:

        Calculates the concordance index (C-index) between two series
        of event times. The first is the real survival times from
        the experimental data, and the other is the predicted survival
        times from a model of some kind.

        The concordance index is a value between 0 and 1 where,
        0.5 is the expected result from random predictions,
        1.0 is perfect concordance and,
        0.0 is perfect anti-concordance (multiply predictions with -1 to get 1.0)

        Score is usually 0.6-0.7 for survival models.

        See:
        Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in
        developing models, evaluating assumptions and adequacy, and measuring and
        reducing errors. Statistics in Medicine 1996;15(4):361-87.
        """
        compute_hazards = theano.function(
            inputs = [self.X],
            outputs = -self.risk_score
        )
        partial_hazards = compute_hazards(data)

        return concordance_index(true_survival,
            partial_hazards,
            event_observed)

    def train(self,
    train_data, valid_data= None,
    standardize = True,
    n_epochs = 500,
    validation_frequency = 10, improvement_threshold = 0.99999,
    patience = 1000, patience_increase = 2,
    update_fn = lasagne.updates.nesterov_momentum,
    verbose = True,
    **kwargs):
        if verbose:
            print '[INFO] Training CoxMLP'

        train_loss = []
        train_ci = []
        x_train, e_train, t_train = train_data['x'], train_data['e'], train_data['t']

        # Sort Training Data
        sort_idx = numpy.argsort(t_train)[::-1]
        x_train = x_train[sort_idx]
        e_train = e_train[sort_idx]
        t_train = t_train[sort_idx]

        if self.standardize:
            self.offset = x_train.mean(axis = 0)
            self.scale = x_train.std(axis = 0)
        # else:
            # self.offset = numpy.zeros_like(x_train[0,:], dtype = numpy.float32)
            # self.scale = numpy.ones_like(x_train[0,:], dtype=numpy.float32)

        # if standardize:
        #     x_mean, x_std = x_train.mean(axis = 0), x_train.std(axis = 0)
        #     x_train = (x_train - x_mean) / x_std

        if valid_data:
            valid_loss = []
            valid_ci = []
            x_valid, e_valid, t_valid = valid_data['x'], valid_data['e'], valid_data['t']

            # Sort Validation Data
            sort_idx = numpy.argsort(t_valid)[::-1]
            x_valid = x_valid[sort_idx]
            e_valid = e_valid[sort_idx]
            t_valid = t_valid[sort_idx]

            # if standardize:
            #     x_valid = (x_valid - x_mean) / x_std

        best_validation_loss = numpy.inf
        best_params = None
        best_params_idx = -1

        lr = theano.shared(numpy.array(self.learning_rate,
                                    dtype = numpy.float32))

        momentum = numpy.array(0, dtype= numpy.float32)

        train_fn, valid_fn = self.get_train_valid_fn(
            L1_reg=self.L1_reg, L2_reg=self.L2_reg,
            learning_rate=lr,
            momentum = momentum,
            update_fn = update_fn, **kwargs
        )

        start = time.time()
        for epoch in xrange(n_epochs):
            # Power-Learning Rate Decay
            lr = self.learning_rate / (1 + epoch * self.lr_decay)

            if self.momentum and epoch >= 10:
                momentum = self.momentum

            loss = train_fn(x_train, e_train)
            train_loss.append(loss)

            ci_train = self.get_concordance_index(
                x_train,
                t_train,
                e_train,
            )
            train_ci.append(ci_train)

            if valid_data and (epoch % validation_frequency == 0):
                validation_loss = valid_fn(x_valid, e_valid)
                valid_loss.append(validation_loss)

                ci_valid = self.get_concordance_index(
                    x_valid,
                    t_valid,
                    e_valid
                )
                valid_ci.append(ci_valid)

                if validation_loss < best_validation_loss:
                    # improve patience if loss improves enough
                    if validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, epoch * patience_increase)

                    best_params = [param.copy().eval() for param in self.params]
                    best_params_idx = epoch
                    best_validation_loss = validation_loss

            if patience <= epoch:
                break

            if epoch % 50 == 0:
                print_progressbar(epoch, n_epochs, verbose = verbose)

        print('Finished Training with %d iterations in %0.2fs' % (
            epoch + 1, time.time() - start
        ))

        metrics = {
            'train': train_loss,
            'best_params': best_params,
            'best_params_idx' : best_params_idx,
            'best_validation_loss':best_validation_loss,
            'train_ci' : train_ci
        }
        if valid_data:
            metrics ['valid'] = valid_loss
            metrics['valid_ci'] = valid_ci
            metrics['best_valid_ci'] = max(valid_ci)

        return metrics
