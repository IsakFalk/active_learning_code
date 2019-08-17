from pathlib import Path

import itertools
import logging

import src.utils.classification as clf_utils
import src.utils.regression as reg_utils

from src.PARAMATERS import img_dir

logging.basicConfig(
    level=logging.DEBUG,
    filename='run-plot_all_learning_curves.log',
    format='%(asctime)s - %(levelname)s # %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


if __name__ == "__main__":

    regression_dirs = [
        'white_wine',
        'red_wine',
        # 'student_performance_math',
        # 'student_performance_port',
        'bike_sharing_day',
        'concrete',
        'boston'
    ]
    classification_dirs = [
        'cencus_income',
        'mnist',
        'yeast'
    ]

    # If directory not in place, create it
    Path(img_dir).mkdir(parents=True, exist_ok=True)

    # Regression plots
    for reg_dir in regression_dirs:
        for traces, plot_type, xlim in itertools.product([False], ['plot'], [[0, 100]]):
            logging.info('Plotting {} with kwargs (traces: {}, plot_type: {}, xlim: {})'.format(
                reg_dir, traces, plot_type, xlim))
            reg_utils.save_learning_curve_k_fold_plot(
                'learning_curves_k_fold-{}'.format(reg_dir),
                traces=traces,
                plot_type=plot_type,
                plot_test=True,
                xlim=xlim)
            reg_utils.save_learning_curve_k_fold_plot(
                'learning_curves_k_fold_realisable-{}'.format(reg_dir),
                traces=traces,
                plot_type=plot_type,
                plot_test=True,
                xlim=xlim)

    # Classification plots
    for clf_dir in classification_dirs:
        for traces, plot_type, xlim in itertools.product([False], ['plot'], [[0, 100]]):
            logging.info('Plotting {} with kwargs (traces: {}, plot_type: {}, xlim: {})'.format(
                reg_dir, traces, plot_type, xlim))
            clf_utils.save_learning_curve_k_fold_plot(
                'learning_curves_k_fold-{}'.format(clf_dir),
                traces=traces,
                plot_type=plot_type,
                plot_test=True,
                xlim=xlim)
