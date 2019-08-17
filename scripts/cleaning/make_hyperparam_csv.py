import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.PARAMATERS import data_experiments_dir, top_dir

"""
Take the hyperparams for all experiments
and put them in a csv file
"""

regression_datasets_dict = {
    "bike_sharing_day": "bike sharing (day)",
    "boston": "boston",
    "concrete": "concrete",
    "red_wine": "red wine",
    "white_wine": "white wine"
}

classification_datasets_dict = {
    "mnist": "mnist",
    "yeast": "yeast"
}


def load_hyperparams(path_to_experiment):
    path_to_json = path_to_experiment / "experiment_config.json"
    with open(path_to_json, 'r') as f:
        json_dict = json.load(f)
        n = json_dict['n']
        d = json_dict['d']
        lambda_opt = json_dict['tau_opt_KRR']
        s_opt = np.sqrt(json_dict['s2_opt_KRR'])

    return n, d, lambda_opt, s_opt


if __name__ == "__main__":
    hyperparams_dir = Path(top_dir) / 'reports' / 'hyperparams'
    hyperparams_dir.mkdir(parents=True, exist_ok=True)

    # Agnostic regression
    data_experiments_dir = Path(data_experiments_dir)
    df_dict = {
        'Dataset': [],
        'n': [],
        'd': [],
        'lambda': [],
        'sigma': []
    }
    for experiment in data_experiments_dir.iterdir():
        dataset_name = experiment.name.split('-')[-1]
        dir_string = experiment.name.split('-')[0]
        if not 'realisable' in dir_string and dataset_name in regression_datasets_dict.keys():
            n, d, lambda_opt, s_opt = load_hyperparams(experiment)
            df_dict['Dataset'].append(
                regression_datasets_dict[dataset_name])
            df_dict['n'].append(n)
            df_dict['d'].append(d)
            df_dict['lambda'].append(lambda_opt)
            df_dict['sigma'].append(s_opt)
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(hyperparams_dir / 'agnostic_regression_hyperparams.csv', index=False, header=True, decimal='.',
              sep=',')

    # Realisable regression
    data_experiments_dir = Path(data_experiments_dir)
    df_dict = {
        'Dataset': [],
        'n': [],
        'd': [],
        'lambda': [],
        'sigma': []
    }
    for experiment in data_experiments_dir.iterdir():
        dataset_name = experiment.name.split('-')[-1]
        dir_string = experiment.name.split('-')[0]
        if 'realisable' in dir_string and dataset_name in regression_datasets_dict.keys():
            n, d, lambda_opt, s_opt = load_hyperparams(experiment)
            df_dict['Dataset'].append(
                regression_datasets_dict[dataset_name])
            df_dict['n'].append(n)
            df_dict['d'].append(d)
            df_dict['lambda'].append(lambda_opt)
            df_dict['sigma'].append(s_opt)
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(hyperparams_dir / 'realisable_regression_hyperparams.csv', index=False, header=True, decimal='.',
              sep=',')

    # Classification
    data_experiments_dir = Path(data_experiments_dir)
    df_dict = {
        'Dataset': [],
        r'n': [],
        r'd': [],
        r'lambda': [],
        r'sigma': []
    }
    for experiment in data_experiments_dir.iterdir():
        dataset_name = experiment.name.split('-')[-1]
        dir_string = experiment.name.split('-')[0]
        if dataset_name in classification_datasets_dict.keys():
            n, d, lambda_opt, s_opt = load_hyperparams(experiment)
            df_dict['Dataset'].append(
                classification_datasets_dict[dataset_name])
            df_dict['n'].append(n)
            df_dict['d'].append(d)
            df_dict['lambda'].append(lambda_opt)
            df_dict['sigma'].append(s_opt)
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(hyperparams_dir / 'agnostic_classification_hyperparams.csv', index=False, header=True, decimal='.',
              sep=',')
