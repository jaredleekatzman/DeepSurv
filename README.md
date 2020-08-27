# DeepSurv

DeepSurv implements a deep learning generalization of the Cox proportional hazards model using Theano and Lasagne. 

DeepSurv has an advantage over traditional Cox regression because it does not require an *a priori* selection of covariates, but learns them adaptively. 

DeepSurv can be used in numerous survival analysis applications. One medical application is provided: recommend_treatment, which provides treatment recommendations for a set of patient observations. 

For more details, see full paper [DeepSurv: Personalized Treatment Recommender System Using A Cox Proportional Hazards Deep Neural Network](http://arxiv.org/abs/1606.00931).

**For an updated implementation of the [Cox loss function](https://github.com/runopti/stg/blob/master/python/stg/losses.py) in PyTorch, please see [Feature Selection using Stochastic Gates (STG) by Yamada et al.](https://github.com/runopti/stg/tree/master/python).**

## Installation:

### From source

Download a local copy of DeepSurv and install from the directory:

	git clone https://github.com/jaredleekatzman/DeepSurv.git
	cd DeepSurv
	pip install .

### Dependencies

Theano, Lasagne (bleeding edge version), lifelines, matplotlib (for visualization), tensorboard_logger, and all of their respective dependencies. 

### Running the tests

After installing, you can optionally run the test suite with

	py.test

from the command line while in the repo's main directory. 

## Running Experiments

Experiments are run using Docker containers built off of the [floydhub](https://github.com/floydhub/dl-docker) deep learning Docker images. DeepSurv can be run on either the CPU or the GPU with nvidia-docker. 

All experiments are in the `DeepSurv/experiments/` directory. 

To run an experiment, define the experiment name as an environmental variable `EXPRIMENT`and run the docker-compose file. Further details are in the `DeepSurv/experiments/` directory. 

## Training a Network

Training DeepSurv can be done in a few lines. 
First, all you need to do is prepare the datasets to have the following keys:

	{ 
		'x': (n,d) observations (dtype = float32), 
	 	't': (n) event times (dtype = float32),
	 	'e': (n) event indicators (dtype = int32)
	}

Then prepare a dictionary of hyper-parameters. And it takes only two lines to train a network:

	network = deepsurv.DeepSurv(**hyperparams)
	log = network.train(train_data, valid_data, n_epochs=500)

You can then evaluate its success on testing data:

	network.get_concordance_index(**test_data)
	>> 0.62269622730138632

If you have matplotlib installed, you can visualize the training and validation curves after training the network:

	deepsurv.plot_log(log)
