# We provide the docker container with a local copy of the DeepSurv module at /DeepSurv/deepsurv
import sys, os
sys.path.append('/deepsurv')

import deep_surv
import datasets
import utils        
from deepsurv_logger import TensorboardLogger

import argparse
import uuid
import pickle
import json

import numpy as np
import lasagne
import optunity

import logging
from logging import handlers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', help='Directory for tensorboard logs')
    parser.add_argument('dataset', help='Dataset to load')
    parser.add_argument('box', help='Filename to box constraints dictionary pickle file')
    parser.add_argument('num_evals', help='Number of models to test', type=int)
    parser.add_argument('--update_fn',help='Lasagne optimizer', default='sgd')
    parser.add_argument('--num_epochs',type=int, help='Number of epochs to train', default=100)
    parser.add_argument('--num_folds', type=int, help='Number of folds to cross-validate', default=5)
    return parser.parse_args()

def load_logger(logdir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Print to Stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    logger.addHandler(ch)

    # Print to Log file
    fh = logging.FileHandler(os.path.join(logdir, 'log_' + str(uuid.uuid4())))
    fh.setFormatter(format)
    logger.addHandler(fh)

    return logger


def load_dataset(dataset):
    '''
    Returns two numpy arrays (x, y) in which:
        x : is the data matrix of (num_examples, num_covariates)
        y : is a two column array containing the censor and time variables for each row in x
    '''
    ## Define internal functions
    def format_to_optunity(dataset, strata=False):
        '''
        Formats a dataset dictionary containing survival data with keys: 
            { 
                'x' : baseline data
                'e' : censor
                't' : event time
            }
        to a format that Optunity can use to run hyper-parameter searches on.
        '''
        x = dataset['x']
        e = dataset['e']
        t = dataset['t']
        y = np.column_stack((e, t))
        # Take the indices of censored entries as strata
        if strata:
            strata = [np.nonzero(np.logical_not(e).astype(np.int32))[0].tolist()]
        else:
            strata = None
        return (x,y,strata)

    def load_simulated_dataset(dataset):
        # Default values
        NUM_EXAMPLES = 5000
        treatment_group = False
        hr_ratio = 5

        # Check if experiment is treatment group        
        if dataset == 'treatment':
            hr_ratio = 10
            treatment_group = True
            dataset = 'gaussian'

        # Generate Data
        factory = datasets.SimulatedData(hr_ratio=hr_ratio,
                average_death=5, 
                censor_method = 'observed_p', observed_p = .5,
                num_features=10, num_var=2,
                treatment_group=treatment_group)
        ds = factory.generate_data(NUM_EXAMPLES, method=dataset)

        return ds

    # Simulated Data Experiments
    if dataset in ['linear', 'gaussian', 'treatment']:
        ds = load_simulated_dataset(dataset)
    else:
        # If not a simulated dataset, load the dataset
        ds = utils.load_datasets(dataset)['train']
    
    return format_to_optunity(ds)

def load_box_constraints(file):
    with open(file, 'rb') as fp:
        return json.loads(fp.read())

def save_call_log(file, call_log):
    with open(file, 'wb') as fp:
        pickle.dump(call_log, fp)

def get_objective_function(num_epochs, logdir, update_fn = lasagne.updates.sgd):
    '''
    Returns the function for Optunity to optimize. The function returned by get_objective_function
    takes the parameters: x_train, y_train, x_test, and y_test, and any additional kwargs to 
    use as hyper-parameters.

    The objective function runs a DeepSurv model on the training data and evaluates it against the
    test set for validation. The result of the function call is the validation concordance index 
    (which Optunity tries to optimize)
    '''
    def format_to_deepsurv(x, y):
        return {
            'x': x,
            'e': y[:,0].astype(np.int32),
            't': y[:,1].astype(np.float32)
        }

    def get_hyperparams(params):
        hyperparams = {
            'batch_norm': False,
            'activation': 'selu',
            'standardize': True
        }
        # @TODO add default parameters and only take necessary args from params
        # protect from params including some other key

        if 'num_layers' in params and 'num_nodes' in params:
            params['hidden_layers_sizes'] = [int(params['num_nodes'])] * int(params['num_layers'])
            del params['num_layers']
            del params['num_nodes']

        if 'learning_rate' in params:
            params['learning_rate'] = 10 ** params['learning_rate']

        hyperparams.update(params)
        return hyperparams

    def train_deepsurv(x_train, y_train, x_test, y_test,
        **kwargs):
        # Standardize the datasets
        train_mean = x_train.mean(axis = 0)
        train_std = x_train.std(axis = 0)

        x_train = (x_train - train_mean) / train_std
        x_test = (x_test - train_mean) / train_std

        train_data = format_to_deepsurv(x_train, y_train)
        valid_data = format_to_deepsurv(x_test, y_test)

        hyperparams = get_hyperparams(kwargs)

        # Set up Tensorboard loggers
        # TODO improve the model_id for Tensorboard to better partition runs
        model_id = str(hash(str(hyperparams)))
        run_id = model_id + '_' + str(uuid.uuid4())
        logger = TensorboardLogger('hyperparam_search', 
            os.path.join(logdir,"tensor_logs", model_id, run_id))

        network = deep_surv.DeepSurv(n_in=x_train.shape[1], **hyperparams)
        metrics = network.train(train_data, n_epochs = num_epochs, logger=logger, 
            update_fn = update_fn, verbose = False)

        result = network.get_concordance_index(**valid_data)
        main_logger.info('Run id: %s | %s | C-Index: %f | Train Loss %f' % (run_id, str(hyperparams), result, metrics['loss'][-1][1]))
        return result

    return train_deepsurv

if __name__ == '__main__':
    args = parse_args()

    NUM_EPOCHS = args.num_epochs
    NUM_FOLDS = args.num_folds

    global main_logger
    main_logger = load_logger(args.logdir)
    main_logger.debug('Parameters: ' + str(args))

    main_logger.debug('Loading dataset: ' + args.dataset)
    x, y, strata = load_dataset(args.dataset)

    box_constraints = load_box_constraints(args.box)
    main_logger.debug('Box Constraints: ' + str(box_constraints))

    opt_fxn = get_objective_function(NUM_EPOCHS, args.logdir, 
        utils.get_optimizer_from_str(args.update_fn))
    opt_fxn = optunity.cross_validated(x=x, y=y, num_folds=NUM_FOLDS,
        strata=strata)(opt_fxn)

    main_logger.debug('Maximizing C-Index. Num_iterations: %d' % args.num_evals)
    opt_params, call_log, _ = optunity.maximize(opt_fxn, num_evals=args.num_evals,
        solver_name='sobol',
        **box_constraints)

    main_logger.debug('Optimal Parameters: ' + str(opt_params))
    main_logger.debug('Saving Call log...')
    print(call_log._asdict())
    save_call_log(os.path.join(args.logdir, 'optunity_log_%s.pkl' % (str(uuid.uuid4()))), call_log._asdict())
    exit(0)

