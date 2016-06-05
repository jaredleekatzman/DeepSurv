# DeepSurv

DeepSurv is a deep learning approach to survival analysis. 


## Installation:

### Dependencies

Theano, Lasagne, lifelines, matplotlib (for visualization) and all of their respective dependencies. 

### Installing

You can install *DeepSurv* using

	pip install deepsurv

from the command line.

## Training a Network

Training DeepSurv can be done in a few lines. 
First, all you need to do is prepare the datasets to have the following keys:

	{ 'x': (n,d) observations, 
	  't': (n) event times,
	  'e': (n) event indicators }

Then prepare a dictionary of hyper-parameters. And it takes only two lines to train a network:

	network = deepsurv.DeepSurv(**hyperparams)
	log = network.train(train_data, valid_data, n_epochs=500)

You can then evaluate its success on testing data:

	network.get_concordance_index(**test_data)
	>> 0.62269622730138632

If you have matplotlib installed, you can visualize the training and validation curves after training the network:

	deepsurv.plot_log(log)



