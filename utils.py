from sklearn import metrics
import numpy as np
import torch
import random
import os
import numpy as np

def basic_metrics(output_probab, label_all):
    auroc = metrics.roc_auc_score(y_true=label_all, y_score=output_probab)
    p_, r_, _ = metrics.precision_recall_curve(y_true=label_all, probas_pred=output_probab)
    auprc = metrics.auc(r_, p_)
    
    f1_list = []
    precision_list = []
    recall_list = []
    accuracy_list = []

    thresholds = np.arange(0.3, 1.0, 0.05)
    for threshold in thresholds:
        output_thresholded = (output_probab > threshold).astype(int)
        precision_list.append(metrics.precision_score(y_true=label_all, y_pred=output_thresholded))
        recall_list.append(metrics.recall_score(y_true=label_all, y_pred=output_thresholded))
        f1_list.append(metrics.f1_score(y_true=label_all, y_pred=output_thresholded))
        accuracy_list.append(metrics.accuracy_score(y_true=label_all, y_pred=output_thresholded))

    best_f1_index = np.argmax(f1_list)
    best_threshold = thresholds[best_f1_index]
    accuracy, f1, p, r = accuracy_list[best_f1_index], f1_list[best_f1_index], precision_list[best_f1_index], recall_list[best_f1_index]
    return auroc, auprc, accuracy, f1, p, r, best_threshold

def calculate_median_auroc(output_probab, label_all, complex_infos_):
    auroc_list = []
    starting_index = 0
    complex_labels = {}
    complex_output_probab = {}

    for complex in complex_infos_:
        c_name = complex['complex_name'].split('.')[0]
        ending_index = complex['total_usable_residues'] + starting_index
        if c_name in complex_labels:
            complex_labels[c_name] = np.concatenate((complex_labels[c_name], label_all[starting_index:ending_index]))
            complex_output_probab[c_name] = np.concatenate((complex_output_probab[c_name], output_probab[starting_index:ending_index]))
        else:
            complex_labels[c_name] = label_all[starting_index:ending_index]
            complex_output_probab[c_name] = output_probab[starting_index:ending_index]
        
        starting_index = ending_index

    for c in complex_labels:
        auroc_list += [metrics.roc_auc_score(y_true=complex_labels[c], y_score=complex_output_probab[c])]
    median_auroc = np.median(auroc_list)
    return median_auroc

def set_seed (seeds=None):
    '''
    :param seeds: a single number or a list of length 5 that contains the following seeds. If a single number is used,
    it will be used for all the random-number-generators in the following parts.
    :return: None
    Applying "manual seeding" for "reproducability" often increases running-time, but it is often a necessary evil.
    (https://stackoverflow.com/questions/56354461/reproducibility-and-performance-in-pytorch)
    '''
    if seeds is None:
        seeds = np.random.randint(low=0, high=1000, size=5)
    if type(seeds) == type(0): # checking whether "seeds" is an integer or not
        seeds = [seeds] * 5
    torch.manual_seed(seeds[0])
    torch.cuda.manual_seed_all(seeds[1])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seeds[2])
    random.seed(seeds[3])
    os.environ['PYTHONHASHSEED'] = str(seeds[4])
    