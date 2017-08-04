'''
Utility functions for visualizing results of DeepSurv experiments
'''


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pylab

import numpy as np

import os

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def extract_value_list(arr):
    return list(np.array(arr)[:,1])

def plot_log(log):
    """
    Plots the training and validation curves for a network's loss function
    and calculated concordance index.

    Parameters:
        log: a dictionary with a list of values for any of the following keys:
            'train': training loss
            'valid': validation loss
            'train_ci': training concordance index
            VALID_CI: validation concordance index
    """
    TRAIN_LOSS = 'loss'
    TRAIN_CI = 'c-index'
    VALID_LOSS = 'valid_loss'
    VALID_CI = 'valid_c-index'

    num_epochs = len(log[TRAIN_LOSS])

    # Plots Negative Log Likelihood vs. Epoch
    fig, ax1 = plt.subplots()
    # plt.figure()
    handles = []
    if TRAIN_LOSS in log:
        epochs = range(num_epochs)
        values = extract_value_list(log[TRAIN_LOSS])
        train, = ax1.plot(epochs, values, 'b', label = 'Training')
        ax1.tick_params('y', colors='b')
        handles.append(train)
    if VALID_LOSS in log:
        ax2 = ax1.twinx()
        epochs = np.linspace(0,num_epochs-1,num=len(log[VALID_LOSS]))
        values = extract_value_list(log[VALID_LOSS])
        valid, = ax2.plot(epochs,values, 'r', label = 'Validation')
        ax2.tick_params('y', colors='r')
        handles.append(valid)
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Likelihood')
    plt.legend(handles=handles, loc = 0)

    # Plots Concordance Index vs. Epoch
    plt.figure()
    handles = []
    if TRAIN_CI in log:
        epochs = np.linspace(0,num_epochs-1,num=len(log[TRAIN_CI]))
        train, = plt.plot(epochs, extract_value_list(log[TRAIN_CI]), label = 'Training')
        handles.append(train)
    if VALID_CI in log:
        epochs = np.linspace(0,num_epochs-1,num=len(log[VALID_CI]))
        valid, = plt.plot(epochs, extract_value_list(log[VALID_CI]), label = 'Validation')
        handles.append(valid)
    plt.xlabel('Epoch')
    plt.ylabel('Concordance Index')
    plt.legend(handles = handles, loc = 4)

def plot_risk_model(x_0, x_1, hr, figsize=(4,3), clim = (-3,3), cmap = 'jet'):
    fig, ax = plt.subplots(figsize=figsize)
    plt.xlim(-1, 1)
    plt.xlabel('$x_0$', fontsize='large')
    plt.xticks(np.arange(-1, 1.5, .5))

    plt.ylim(-1, 1)
    plt.ylabel('$x_1$', fontsize='large')
    plt.yticks(np.arange(-1, 1.5, .5))
    
    im = plt.scatter(x=x_0, y=x_1, c=hr, marker='.', cmap=cmap)
    fig.colorbar(im)
    # plt.clim(0, 1)
    plt.clim(*clim)
    plt.tight_layout()
    return (fig, ax, im)

def save_fig(fig, fp):
    # TODO fit the pdf saving cutting off the x and y axis labels
    pp_true = PdfPages(fp)
    pp_true.savefig(fig, dpi=600)
    pp_true.close()

def plot_experiment_scatters(risk_fxn, dataset, norm_vals = None, output_file=None, 
    figsize = (4,3), clim=(-3,3), cmap = 'jet', plot_error=False, trt_idx = None):
    
    def norm_hr(hr):
        # return hr
        return hr - hr.mean();
        # return (hr - hr.min()) / (hr.max() - hr.min())

    x_0 = dataset['x'][:, 0]
    x_1 = dataset['x'][:, 1]

    # Plot model predictions
    x = dataset['x']
    if norm_vals:
        x = (x - norm_vals['mean']) / norm_vals['std']

    (head, tail) = os.path.split(output_file)

    if not trt_idx is None:
        trt_values = np.unique(x[:,trt_idx])
        for (idx,trt_value) in enumerate(trt_values):
            x_trt = np.copy(x)
            x_trt[:,trt_idx] = trt_value
            hr_trt = risk_fxn(x_trt)
            hr_trt = norm_hr(hr_trt)
            fig_trt, _, _ = plot_risk_model(x_0, x_1, hr_trt, figsize, clim, cmap)

            if output_file:
                save_fig(fig_trt, os.path.join(head, "treatment_%d_" % idx + tail))
    else:
        hr_pred = risk_fxn(x)
        hr_pred = norm_hr(hr_pred)
        fig_pred, _, _ = plot_risk_model(x_0, x_1, hr_pred, figsize, clim, cmap)

        if output_file:
            save_fig(fig_pred, os.path.join(head, "pred_" + tail))

    if 'hr' in dataset:
        hr_true = dataset['hr']
        hr_true = norm_hr(hr_true)
        fig_true, _, _ = plot_risk_model(x_0, x_1, hr_true, figsize, clim, cmap)

        if output_file:
            save_fig(fig_true, os.path.join(head, "true_" + tail))

        if plot_error:
            hr_error = np.abs(hr_true - hr_pred)
            fig_error, _, _ = plot_risk_model(x_0, x_1, hr_error, figsize, clim=(0,20), cmap = cmap)

            if output_file:
                save_fig(fig_error, os.path.join(head, "error_" + tail))

def plot_survival_curves(rec_t, rec_e, antirec_t, antirec_e, experiment_name = '', output_file = None):
    # Set-up plots
    plt.figure(figsize=(12,3))
    ax = plt.subplot(111)

    # Fit survival curves
    kmf = KaplanMeierFitter()
    kmf.fit(rec_t, event_observed=rec_e, label=' '.join([experiment_name, "Recommendation"]))   
    kmf.plot(ax=ax,linestyle="-")
    kmf.fit(antirec_t, event_observed=antirec_e, label=' '.join([experiment_name, "Anti-Recommendation"]))
    kmf.plot(ax=ax,linestyle="--")
    
    # Format graph
    plt.ylim(0,1);
    ax.set_xlabel('Timeline (months)',fontsize='large')
    ax.set_ylabel('Percentage of Population Alive',fontsize='large')
    
    # Calculate p-value
    results = logrank_test(rec_t, antirec_t, rec_e, antirec_e, alpha=.95)
    results.print_summary()

    # Location the label at the 1st out of 9 tick marks
    xloc = max(np.max(rec_t),np.max(antirec_t)) / 9
    if results.p_value < 1e-5:
        ax.text(xloc,.2,'$p < 1\mathrm{e}{-5}$',fontsize=20)
    else:
        ax.text(xloc,.2,'$p=%f$' % results.p_value,fontsize=20)
    plt.legend(loc='best',prop={'size':15})


    if output_file:
        plt.tight_layout()
        pylab.savefig(output_file)
