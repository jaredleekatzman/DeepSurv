import argparse
from collections import defaultdict

import pandas as pd
import numpy as np
import h5py
import uuid
import copy
import json

import sys, os
sys.path.append("/DeepSurv/deepsurv")
import deep_surv


# Force matplotlib to not use any Xwindows backend.
import matplotlib
matplotlib.use('Agg')

import viz
import utils
from deepsurv_logger import TensorboardLogger

import time
localtime   = time.localtime()
TIMESTRING  = time.strftime("%m%d%Y%M", localtime)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', help="name of the experiment that is being run")
    parser.add_argument('model', help='Model .json file to load')
    parser.add_argument('dataset', help='.h5 File containing the train/valid/test datasets')
    parser.add_argument('--update_fn',help='Which lasagne optimizer to use (ie. sgd, adam, rmsprop)', default='sgd')
    parser.add_argument('--plot_error', action="store_true", help="If arg present, plot absolute error plots")
    parser.add_argument('--treatment_idx', default=None, type=int, help='(Optional) column index of treatment variable in dataset. If present, run treatment visualizations.')
    parser.add_argument('--results_dir', help="Output directory to save results (model weights, visualizations)", default=None)
    parser.add_argument('--weights', help='(Optional) Weights .h5 File', default=None)
    parser.add_argument('--num_epochs', type=int, default=500, help="Number of epochs to train for. Default: 500")
    return parser.parse_args()

def evaluate_model(model, dataset, bootstrap = False):
    def mse(model):
        def deepsurv_mse(x, hr, **kwargs):
            hr_pred = np.squeeze(model.predict_risk(x))
            return ((hr_pred - hr) ** 2).mean()

        return deepsurv_mse

    metrics = {}

    # Calculate c_index
    metrics['c_index'] = model.get_concordance_index(**dataset)
    if bootstrap:
        metrics['c_index_bootstrap'] = utils.bootstrap_metric(model.get_concordance_index, dataset)
    
    # Calcualte MSE
    if 'hr' in dataset:
        metrics['mse'] = mse(model)(**dataset)
        if bootstrap:
            metrics['mse_bootstrap'] = utils.bootstrap_metric(mse(model), dataset)

    return metrics

def save_risk_surface_visualizations(model, dataset, norm_vals, output_dir, plot_error, experiment,
    trt_idx):
    if experiment == 'linear':
        clim = (-3,3)
    elif experiment == 'gaussian' or experiment == 'treatment':
        clim = (-1,1)
    else:
        clim = (0,1)

    risk_fxn = lambda x: np.squeeze(model.predict_risk(x))
    color_output_file = os.path.join(output_dir, "deep_viz_color_" + TIMESTRING + ".pdf")
    viz.plot_experiment_scatters(risk_fxn, dataset, norm_vals = norm_vals, 
        output_file=color_output_file, figsize=(4,3), clim=clim, 
        plot_error = plot_error, trt_idx = trt_idx)

    bw_output_file = os.path.join(output_dir, "deep_viz_bw_" + TIMESTRING + ".pdf")
    viz.plot_experiment_scatters(risk_fxn, dataset, norm_vals = norm_vals, 
        output_file=bw_output_file, figsize=(4,3), clim=clim, cmap='gray', 
        plot_error = plot_error, trt_idx = trt_idx)


def save_treatment_rec_visualizations(model, dataset, output_dir, 
    trt_i = 1, trt_j = 0, trt_idx = 0):
    
    trt_values = np.unique(dataset['x'][:,trt_idx])
    print("Recommending treatments:", trt_values)
    rec_trt = model.recommend_treatment(dataset['x'], trt_i, trt_j, trt_idx)
    rec_trt = np.squeeze((rec_trt < 0).astype(np.int32))

    rec_dict = utils.calculate_recs_and_antirecs(rec_trt, true_trt = trt_idx, dataset = dataset)
    
    output_file = os.path.join(output_dir, '_'.join(['deepsurv',TIMESTRING, 'rec_surv.pdf']))
    print(output_file)
    viz.plot_survival_curves(experiment_name = 'DeepSurv', output_file=output_file, **rec_dict)

def save_model(model, output_file):
    model.save_weights(output_file)

if __name__ == '__main__':
    args = parse_args()
    print("Arguments:",args)

    # Load Dataset
    print("Loading datasets: " + args.dataset)
    datasets = utils.load_datasets(args.dataset)
    norm_vals = {
            'mean' : datasets['train']['x'].mean(axis =0),
            'std'  : datasets['train']['x'].std(axis=0)
        }

    # Train Model
    tensor_log_dir = "/shared/data/logs/tensorboard_" + str(args.dataset) + "_" + str(uuid.uuid4())
    logger = TensorboardLogger("experiments.deep_surv", tensor_log_dir, update_freq = 10)
    model = deep_surv.load_model_from_json(args.model, args.weights)
    if 'valid' in datasets:
        valid_data = datasets['valid']
    else:
        valid_data = None
    metrics = model.train(datasets['train'], valid_data, n_epochs = args.num_epochs, logger=logger,
        update_fn = utils.get_optimizer_from_str(args.update_fn),
        validation_frequency = 100)

    # Evaluate Model
    with open(args.model, 'r') as fp:
        json_model = fp.read()
        hyperparams = json.loads(json_model)

    train_data = datasets['train']
    if hyperparams['standardize']:
        train_data = utils.standardize_dataset(train_data, norm_vals['mean'], norm_vals['std'])

    metrics = evaluate_model(model, train_data)
    print("Training metrics: " + str(metrics))
    if 'valid' in datasets:
        valid_data = datasets['valid']
        if hyperparams['standardize']:
            valid_data = utils.standardize_dataset(valid_data, norm_vals['mean'], norm_vals['std'])
            metrics = evaluate_model(model, valid_data)
        print("Valid metrics: " + str(metrics))

    if 'test' in datasets:
        test_dataset = utils.standardize_dataset(datasets['test'], norm_vals['mean'], norm_vals['std'])
        metrics = evaluate_model(model, test_dataset, bootstrap=True)
        print("Test metrics: " + str(metrics))

    if 'viz' in datasets:
        print("Saving Visualizations")
        save_risk_surface_visualizations(model, datasets['viz'], norm_vals = norm_vals,
            output_dir=args.results_dir, plot_error = args.plot_error, 
            experiment = args.experiment, trt_idx= args.treatment_idx)

    if 'test' in datasets and args.treatment_idx is not None:
        print("Calculating treatment recommendation survival curvs")
        # We use the test dataset because these experiments don't have a viz dataset
        save_treatment_rec_visualizations(model, test_dataset, output_dir=args.results_dir, 
            trt_idx = args.treatment_idx)

    if args.results_dir:
        _, model_str = os.path.split(args.model)
        output_file = os.path.join(args.results_dir,"models") + model_str + str(uuid.uuid4()) + ".h5"
        print("Saving model parameters to output file", output_file)
        save_model(model, output_file)

    exit(0)

