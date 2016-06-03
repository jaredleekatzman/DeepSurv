import matplotlib
import matplotlib.pyplot as plt

import theano
import numpy as np


def plot_log(log):
    """
    Plots the training and validation curves for a network's loss function
    and calculated concordance index.

    Parameters:
        log: a dictionary with a list of values for any of the following keys:
            'train': training loss
            'valid': validation loss
            'train_ci': training concordance index
            'valid_ci': validation concordance index
    """
    # Plots Negative Log Likelihood vs. Epoch
    plt.figure()
    handles = []
    if 'train' in log:
        epochs = range(len(log['train']))
        train, = plt.plot(epochs,log['train'], label = 'Training')
        handles.append(train)
    if 'valid' in log:
        epochs = np.linspace(0,len(log['train'])-1,num=len(log['valid']))
        valid, = plt.plot(epochs,log['valid'], label = 'Validation')
        handles.append(valid)
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Likelihood')
    plt.legend(handles=handles, loc = 0)

    # Plots Concordance Index vs. Epoch
    plt.figure()
    handles = []
    if 'train_ci' in log:
        epochs = np.linspace(0,len(log['train'])-1,num=len(log['train_ci']))
        train, = plt.plot(epochs, log['train_ci'], label = 'Training')
        handles.append(train)
    if 'valid_ci' in log:
        epochs = np.linspace(0,len(log['train'])-1,num=len(log['valid_ci']))
        valid, = plt.plot(epochs, log['valid_ci'], label = 'Validation')
        handles.append(valid)
    plt.xlabel('Epoch')
    plt.ylabel('Concordance Index')
    plt.legend(handles = handles, loc = 4)
