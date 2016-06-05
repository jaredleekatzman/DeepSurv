DeepSurv
========

DeepSurv implements a deep learning generalization of the Cox proportional hazards model using Theano and Lasagne. 

DeepSurv has an advantage over traditional Cox regression because it does not require an *a priori* selection of covariates, but learns them adaptively. 

DeepSurv can be used in numerous survival analysis applications. One medical application is provided, recommend_treatment, which provides treatment recommendations for a set of patient observations. 

Installation:
-------------

Dependencies:
^^^^^^^^^^^^^

Theano, Lasagne, lifelines, matplotlib (for visualization) and all of their respective dependencies. 

Installing:
^^^^^^^^^^^

You can install *DeepSurv* using

::

	pip install deepsurv

from the command line.

Running the tests:
^^^^^^^^^^^^^^^^^^
After installing, you can optionally run the test suite with

::

	py.test

from the command line while in the module main directory.

Training a Network:
-------------------

Training DeepSurv can be done in a few lines. 
First, all you need to do is prepare the datasets to have the following keys:

::

	{ 
		'x': (n,d) observations (dtype = float32), 
	 	't': (n) event times (dtype = float32),
	 	'e': (n) event indicators (dtype = int32)
	}

Then prepare a dictionary of hyper-parameters. And it takes only two lines to train a network:

::

	network = deepsurv.DeepSurv(**hyperparams)
	log = network.train(train_data, valid_data, n_epochs=500)

You can then evaluate its success on testing data:

::

	network.get_concordance_index(**test_data)
	>> 0.62269622730138632

If you have matplotlib installed, you can visualize the training and validation curves after training the network:

::

	deepsurv.plot_log(log)



License:
-------- 

MIT License

Copyright (c) 2016, Jared Katzman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
