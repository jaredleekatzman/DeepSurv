import sys, os
sys.path.append("/DeepSurv/deepsurv")

# Force matplotlib to not use any Xwindows backend.
import matplotlib
matplotlib.use('Agg')
import viz
import utils

import argparse
from collections import defaultdict

import pandas as pd
import numpy as np
import h5py

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

import logging

import uuid
import time
localtime   = time.localtime()
TIMESTRING  = time.strftime("%m%d%Y%M", localtime)

DURATION_COL = 'time'
EVENT_COL = 'censor'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', help='name of the experiment that is being run')
    parser.add_argument('dataset', help='.h5 File containing the train/valid/test datasets')
    parser.add_argument('--results_dir', default='/shared/results', help='Directory to save resulting models and visualizations')
    parser.add_argument('--plot_error', action="store_true", help="If arg present, plot absolute error plots")
    parser.add_argument('--treatment_idx', default=None, type=int, help='(Optional) column index of treatment variable in dataset. If present, run treatment visualizations.')
    return parser.parse_args()

def evaluate_model(model, dataset, bootstrap = False):
    def ci(model):
        def cph_ci(x, t, e, **kwargs):
            return concordance_index(
                event_times= t, 
                predicted_event_times= -model.predict_partial_hazard(x), 
                event_observed= e,
            )
        return cph_ci

    def mse(model):
        def cph_mse(x, hr, **kwargs):
            hr_pred = np.squeeze(-model.predict_partial_hazard(x).values)
            return ((hr_pred - hr) ** 2).mean()
        return cph_mse  

    metrics = {}

    # Calculate c_index
    metrics['c_index'] = ci(model)(**dataset)
    if bootstrap:
        metrics['c_index_bootstrap'] = utils.bootstrap_metric(ci(model), dataset)
    
    # Calcualte MSE
    if 'hr' in dataset:
        metrics['mse'] = mse(model)(**dataset)
        if bootstrap:
            metrics['mse_bootstrap'] = utils.bootstrap_metric(mse(model), dataset)

    return metrics

def save_visualizations(model, dataset, output_dir, plot_error, experiment):
    if experiment == 'linear':
        clim = (-3,3)
    elif experiment == 'gaussian':
        clim = (-1,1)
    else:
        clim = (0,1)

    risk_fxn = lambda x: np.squeeze(model.predict_partial_hazard(x))
    color_output_file = os.path.join(output_dir, "cph_viz_color_" + TIMESTRING + ".pdf")
    viz.plot_experiment_scatters(risk_fxn, dataset, 
        output_file=color_output_file, figsize=(4,3), clim=clim, plot_error=plot_error)
    # Print BW
    bw_output_file = os.path.join(output_dir, "cph_viz_bw_" + TIMESTRING + ".pdf")
    viz.plot_experiment_scatters(risk_fxn, dataset, 
        output_file=bw_output_file, figsize=(4,3), clim=clim, cmap='gray', plot_error=plot_error)


def save_treatment_rec_visualizations(model, dataset, output_dir, 
    trt_i = 1, trt_j = 0, trt_idx = 0):
    rec_trt = model.recommend_treatment(dataset['x'], trt_i, trt_j, trt_idx)
    rec_trt = np.squeeze((rec_trt < 0).astype(np.int32))

    rec_dict = utils.calculate_recs_and_antirecs(rec_trt, true_trt = trt_idx, dataset = dataset)
    
    output_file = os.path.join(output_dir, '_'.join(['deepsurv',TIMESTRING, 'rec_surv.pdf']))
    print(output_file)
    viz.plot_survival_curves(experiment_name = 'DeepSurv', output_file=output_file, **rec_dict)

def save_treatment_rec_visualizations(model, dataset, output_dir, trt_idx = None):
    tds = utils.format_dataset_to_df(dataset, DURATION_COL, EVENT_COL, trt_idx = trt_idx)

    rec_trt = rsf_treatment_rec(model,tds)
    # rec_trt = (rec_trt < 0).astype(np.int32)

    rec_dict = utils.calculate_recs_and_antirecs(rec_trt, true_trt = trt_idx, dataset = dataset)

    output_file = os.path.join(output_dir, '_'.join(['rsf', TIMESTRING,'rec_surv.pdf']))
    viz.plot_survival_curves(experiment_name = 'RSF', output_file=output_file, **rec_dict)

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    args = parse_args()
    print("Arguments:",args)

    # Load Dataset
    print("Loading datasets: " + args.dataset)
    datasets = utils.load_datasets(args.dataset)

    # Train CPH model
    print("Training CPH Model")
    train_df = utils.format_dataset_to_df(datasets['train'], DURATION_COL, EVENT_COL)
    cf = CoxPHFitter()
    results = cf.fit(train_df, duration_col=DURATION_COL, event_col=EVENT_COL, 
        include_likelihood=True)
    cf.print_summary()
    print("Train Likelihood: " + str(cf._log_likelihood))

    if 'valid' in datasets:
        metrics = evaluate_model(cf, datasets['valid'])
        print("Valid metrics: " + str(metrics))

    if 'test' in datasets:
        metrics = evaluate_model(cf, datasets['test'], bootstrap=True)
        print("Test metrics: " + str(metrics))

    print("Saving Visualizations")
    if 'test' in datasets and args.treatment_idx is not None:
        print("Calculating treatment recommendation survival curvs")
        # We use the test dataset because these experiments don't have a viz dataset
        save_treatment_rec_visualizations(model, test_dataset, output_dir=args.results_dir, 
            trt_idx = args.treatment_idx)

    if 'viz' in datasets:
        save_visualizations(cf, datasets['viz'], output_dir=args.results_dir, plot_error = args.plot_error,
            experiment = args.experiment)

    exit(0)

