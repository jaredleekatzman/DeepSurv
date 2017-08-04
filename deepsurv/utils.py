'''
Utility functions for running DeepSurv experiments
'''

import h5py
import scipy.stats as st
from collections import defaultdict
import numpy as np
import copy

import lasagne

def load_datasets(dataset_file):
    datasets = defaultdict(dict)

    with h5py.File(dataset_file, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]

    return datasets

def format_dataset_to_df(dataset, duration_col, event_col, trt_idx = None):
    xdf = pd.DataFrame(dataset['x'])
    if trt_idx is not None:
        xdf = xdf.rename(columns={trt_idx : 'treat'})

    dt = pd.DataFrame(dataset['t'], columns=[duration_col])
    censor = pd.DataFrame(dataset['e'], columns=[event_col])
    cdf = pd.concat([xdf, dt, censor], axis=1)
    return cdf

def standardize_dataset(dataset, offset, scale):
    norm_ds = copy.deepcopy(dataset)
    norm_ds['x'] = (norm_ds['x'] - offset) / scale
    return norm_ds

def bootstrap_metric(metric_fxn, dataset, N=100):
    def sample_dataset(dataset, sample_idx):
        sampled_dataset = {}
        for (key,value) in dataset.items():
            sampled_dataset[key] = value[sample_idx]
        return sampled_dataset

    metrics = []
    size = len(dataset['x'])

    for _ in range(N):
        resample_idx = np.random.choice(size, size=size, replace = True)
    
        metric = metric_fxn(**sample_dataset(dataset, resample_idx))
        metrics.append(metric)
    
    # Find mean and 95% confidence interval
    mean = np.mean(metrics)
    conf_interval = st.t.interval(0.95, len(metrics)-1, loc=mean, scale=st.sem(metrics))
    return {
        'mean': mean,
        'confidence_interval': conf_interval
    }

def get_optimizer_from_str(update_fn):
    if update_fn == 'sgd':
        return lasagne.updates.sgd
    elif update_fn == 'adam':
        return lasagne.updates.adam
    elif update_fn == 'rmsprop':
        return lasagne.updates.rmsprop

    return None

def calculate_recs_and_antirecs(rec_trt, true_trt, dataset, print_metrics=True):
    if isinstance(true_trt, int):
        true_trt = dataset['x'][:,true_trt]

    # trt_values = zip([0,1],np.sort(np.unique(true_trt)))
    trt_values = enumerate(np.sort(np.unique(true_trt)))
    equal_trt = [np.logical_and(rec_trt == rec_value, true_trt == true_value) for (rec_value, true_value) in trt_values]
    rec_idx = np.logical_or(*equal_trt)
    # original Logic
    # rec_idx = np.logical_or(np.logical_and(rec_trt == 1,true_trt == 1),
    #               np.logical_and(rec_trt == 0,true_trt == 0))

    rec_t = dataset['t'][rec_idx]
    antirec_t = dataset['t'][~rec_idx]
    rec_e = dataset['e'][rec_idx]
    antirec_e = dataset['e'][~rec_idx]

    if print_metrics:
        print("Printing treatment recommendation metrics")
        metrics = {
            'rec_median' : np.median(rec_t),
            'antirec_median' : np.median(antirec_t)
        }
        print("Recommendation metrics:", metrics)

    return {
        'rec_t' : rec_t, 
        'rec_e' : rec_e, 
        'antirec_t' : antirec_t, 
        'antirec_e' : antirec_e
    }
    

