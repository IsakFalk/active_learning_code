import argparse
import logging
import sys

import numpy as np

import src.utils.classification as clf_utils
import src.utils.load_functions as load_functions
import src.utils.regression as reg_utils
from src.utils.utils import subsample_dataset

# Seed rng to make reproducible
np.random.seed(123)


"""Calculate all of the learning curves in one go
and save these in a pandas dataframe (HD5)

This will be updated as we go along."""

logging.basicConfig(
    level=logging.DEBUG,
    filename='run-calculate_learning_curves.log',
    format='%(asctime)s - %(levelname)s # %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_datapoints",
        "-n",
        type=int,
        help="Size of dataset, if this is smaller than original dataset we subsample",
    )
    args = parser.parse_args()

    num_datapoints = args.num_datapoints
    logging.info(
        'Maximum dataset size set to {} points'.format(num_datapoints))

    # str(name) => load_function
    datasets_regression = {
        'boston': load_functions.load_boston,
        'white_wine': load_functions.load_white_wine,
        'red_wine': load_functions.load_red_wine,
        'student_performance_math': load_functions.load_student_performance_math,
        'student_performance_port': load_functions.load_student_performance_port,
        'bike_sharing_day': load_functions.load_bike_sharing_day,
        'concrete': load_functions.load_concrete
    }
    datasets_classificiation = {
        'cencus_income': load_functions.load_cencus_income,
        'mnist': load_functions.load_mnist,
        'yeast': load_functions.load_yeast
    }

    # Normalise all continuous regressors (on whole dataset, so slightly cheating)
    normalise = True
    # Perform regression calculation
    for name, load_func in datasets_regression.items():
        logging.info('Running dataset: {}'.format(name))
        X, y = load_func(normalise=normalise)
        X, y = subsample_dataset(X, y, num_datapoints)
        reg_utils.run_learning_curve_experiment_k_fold(X, y, dataset_name=name)
        logging.info('Done')

    # Perform classification calculation
    for name, load_func in datasets_classificiation.items():
        logging.info('Running dataset: {}'.format(name))
        X, y = load_func(normalise=normalise)
        X, y = subsample_dataset(X, y, num_datapoints)
        clf_utils.run_learning_curve_experiment_k_fold(X, y, dataset_name=name)
        logging.info('Done')

    logging.info('All datasets finished.')
