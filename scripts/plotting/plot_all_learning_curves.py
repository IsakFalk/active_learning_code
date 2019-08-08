import itertools
import logging

import src.utils.classification as clf_utils
import src.utils.regression as reg_utils

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
        'student_performance_math',
        'student_performance_port',
        'bike_sharing_day',
        'concrete'
    ]
    classification_dirs = [
        'cencus_income',
        'mnist',
        'yeast'
    ]

    # Regression plots
    for reg_dir in regression_dirs:
        for traces, plot_type in itertools.product([False], ['plot']):
            logging.info('Plotting {} with kwargs (traces: {}, plot_type: {})'.format(
                reg_dir, traces, plot_type))
            reg_utils.save_learning_curve_k_fold_plot(
                'learning_curves_k_fold-{}'.format(reg_dir), traces=traces, plot_type=plot_type)
            reg_utils.save_learning_curve_k_fold_plot(
                'learning_curves_k_fold_realisable-{}'.format(reg_dir), traces=traces, plot_type=plot_type)

    # Classification plots
    for clf_dir in classification_dirs:
        for traces, plot_type in itertools.product([False], ['plot']):
            logging.info('Plotting {} with kwargs (traces: {}, plot_type: {})'.format(
                reg_dir, traces, plot_type))
            clf_utils.save_learning_curve_k_fold_plot(
                'learning_curves_k_fold-{}'.format(clf_dir), traces=traces, plot_type=plot_type)
