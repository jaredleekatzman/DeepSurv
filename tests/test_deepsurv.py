
import pytest

import deepsurv
from deepsurv import DeepSurv

import numpy

def generate_data(treatment_group = False):
    numpy.random.seed(123)
    sd = deepsurv.datasets.SimulatedData(5, num_features = 9,
        treatment_group = treatment_group)
    train_data = sd.generate_data(5000)
    valid_data = sd.generate_data(2000)
    test_data = sd.generate_data(2000)

    return train_data, valid_data, test_data


class TestDeepSurvInit():

    @classmethod
    def setup_class(self):
        self.hidden_layers_sizes = [10,10]
        self.hyperparams = {
            'n_in': 10,
            'learning_rate': 1e-5,
            'hidden_layers_sizes': self.hidden_layers_sizes
        }


    def test_deepsurv_initialize_layers(self):
        network = DeepSurv(**self.hyperparams)

        # Hidden Layers + 1 output layer
        assert len(network.hidden_layers) == len(self.hidden_layers_sizes) + 1

    def test_deepsurv_initialize_batch_norm(self):
        network = DeepSurv(batch_norm = True, **self.hyperparams)
        assert len(network.hidden_layers) == 3 * len(self.hidden_layers_sizes) + 1

    def test_deepsurv_initialize_dropout(self):
        network = DeepSurv(dropout = 0.5, **self.hyperparams)
        assert len(network.hidden_layers) == 2 * len(self.hidden_layers_sizes) + 1

    def test_deepsurv_initialize_standardize_layer(self):
        network = DeepSurv(standardize = True, **self.hyperparams)
        assert len(network.hidden_layers) == len(self.hidden_layers_sizes) + 3

class TestDeepSurvTrain():

    @classmethod
    def setup_class(self):
        self.train, self.valid, self.test = generate_data(treatment_group=True)

        hyperparams = {
            'n_in': 10,
            'learning_rate': 1e-5,
            'hidden_layers_sizes': [10]
        }
        network = DeepSurv(**hyperparams)
        log = network.train(self.train, self.valid,
            n_epochs=10,validation_frequency=1)
        self.log = log
        self.network = network

    def test_train(self):
        # Test if network computes nan for any values
        assert self.log.has_key('train') == True
        assert numpy.any(numpy.isnan(self.log['train'])) == False

        assert self.log.has_key('valid') == True
        assert numpy.any(numpy.isnan(self.log['valid'])) == False

    def test_network_predict_risk(self):
        risk = self.network.predict_risk(self.test['x'])

        assert numpy.any(numpy.isnan(risk)) == False

    def test_get_concordance_index(self):
        ci = self.network.get_concordance_index(**self.test)
        assert ci >= 0.0

    def test_recommend_treatment(self):
        x = self.test['x']
        trt_0, trt_1 = numpy.unique(self.train['x'][:,-1])
        rec = self.network.recommend_treatment(x,trt_0,trt_1)

        assert numpy.any(numpy.isnan(rec)) == False
