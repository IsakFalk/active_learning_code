import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import src.algorithms as alg
import src.visualization.visualize as viz
from src.PARAMATERS import data_experiments_dir, img_dir
from src.utils.utils import (create_krr_mmd_kernel_matrices,
                             gaussian_kernel_matrix, k_fold_train_test_indices,
                             kernel_quantile_heuristic)


def pick_optimal_params_using_cv(X, y, cv=5, iid=False):
    """Using cross validation on dataset, pick best parameters

    Using the gaussian kernel ridge regression for classifiction, pick the best tau and s2
    using cross validation on the data. Internally use the 0.05 and 0.95
    quantile as base low and high values to linspace over.

    :param X: (np.ndarray, (n, d)) design matrix
    :param y: (np.ndarray, (n, 1)) output array
    :param cv: (int) number of folds in cross-validation
    :param iid: (bool) if we assume iid or not (False lead to normal k-fold cv)

    :return gkrclf_cv: (GaussianKRRClassification) optimised gkrclf instance through sklearn GridSearchCV
    :rerurn tau: (float) optimal tau for GKRR
    :return s2: (float) optimal s2 for GKRR"""
    gaussian_krc = GaussianKRRClassification()

    q10_sq_dist = kernel_quantile_heuristic(X, q=0.05)
    q90_sq_dist = kernel_quantile_heuristic(X, q=0.95)

    param_grid = dict(
        tau=np.logspace(-3, 1, 5),
        s2=np.linspace(q10_sq_dist * 0.1, q90_sq_dist * 10, 5)
    )

    gkrc_cv = GridSearchCV(estimator=gaussian_krc,
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=cv, iid=iid)
    gkrc_cv.fit(X, y)
    tau = gkrc_cv.best_params_['tau']
    s2 = gkrc_cv.best_params_['s2']
    return gkrc_cv, tau, s2


def calculate_learning_curves_train_test(K, y, train_indices, test_indices, sampled_order_train,
                                         tau, stop_t=None):
    """Calculate learning curves (train, test) from running herding algorithm

    Using the sampled order from the sampled_order indexing array
    calculate the learning curves on the train set using GKRR. Note that we
    pass K instead of calculating it on the fly, that's why we don't use
    s2 explicitly, it's already used in calculating K.

    :param K: (np.ndarray, (n, n)) full kernel matrix from dataset
    :param y: (np.ndarray, (n, 1)) output array
    :param train_indices: (np.ndarray, (n_train,)) train indices from the original dataset
    :param test_indices: (np.ndarray, (n_train,)) test indices from the original dataset
    :param sampled_order_train: (np.ndarray, (n_train,)) order of the sampled training indices
    :param tau: (float) regularisation parameter used in GKRR
    :param stop_t: (int) final step of calculations

    :return learning_curve_train: (np.ndarray, (stop_t,)) array of mse for train set
    :return learning_curve_test: (np.ndarray, (stop_t,)) array of mse for test set
    """
    gaussian_krc = GaussianKRRClassification(
        tau=tau, s2=None, precompute_K=True)

    # Index K differently depending on what we do.
    # When predicting, we need the kernel matrix to be
    # K_mn, where m indexes the set to predict over and
    # n indexes the set we train over
    K_train = K[np.ix_(train_indices, train_indices)]
    K_test = K[np.ix_(test_indices, test_indices)]
    K_test_train = K[np.ix_(test_indices, train_indices)]
    K_sampled_train = K_train[np.ix_(sampled_order_train, sampled_order_train)]

    y_train = y[train_indices]
    y_test = y[test_indices]
    y_sampled_train = y_train[sampled_order_train]

    n_train = K_train.shape[0]
    n_test = K_test.shape[0]

    if stop_t is None:
        stop_t = n_train

    learning_curve_train = np.zeros(stop_t)
    learning_curve_test = np.zeros(stop_t)

    for t in range(stop_t):
        K_sampled_train_t = K_sampled_train[0:t+1, 0:t+1]
        gaussian_krc.fit(X=K_sampled_train_t, y=y_sampled_train[:t+1])

        # Predict for train set
        K_xn_train = K_train[np.ix_(
            np.arange(n_train), sampled_order_train[:t+1])]
        y_train_ = gaussian_krc.predict(K_xn_train)
        learning_curve_train[t] = accuracy_score(y_train, y_train_)

        # Then test set
        K_xn_test = K_test_train[np.ix_(
            np.arange(n_test), sampled_order_train[:t+1])]
        y_test_ = gaussian_krc.predict(K_xn_test)
        learning_curve_test[t] = accuracy_score(y_test, y_test_)

    return learning_curve_train, learning_curve_test


def calculate_learning_curves(K, y, train_indices, test_indices, sampled_order_train,
                              tau, stop_t=None):
    """Calculate learning curves (train, test) from running a herding algorithm

    Using the sampled order from the sampled_order indexing array
    calculate the learning curves on the train set using GKRR. Note that we
    pass K instead of calculating it on the fly, that's why we don't use
    s2 explicitly, it's already used in calculating K.

    :param K: (np.ndarray, (n, n)) full kernel matrix from dataset
    :param y: (np.ndarray, (n, 1)) output array
    :param train_indices: (np.ndarray, (n_train,)) train indices from the original dataset
    :param test_indices: (np.ndarray, (n_train,)) test indices from the original dataset
    :param sampled_order_train: (np.ndarray, (n_train,)) order of the sampled training indices
    :param tau: (float) regularisation parameter used in GKRR
    :param stop_t: (int) final step of calculations

    :return learning_curve_train: (np.ndarray, (stop_t,)) array of mse for train set
    :return learning_curve_test: (np.ndarray, (stop_t,)) array of mse for test set
    """
    gaussian_krc = GaussianKRRClassification(
        tau=tau, s2=None, precompute_K=True)

    # Index K differently depending on what we do.
    # When predicting, we need the kernel matrix to be
    # K_mn, where m indexes the set to predict over and
    # n indexes the set we train over
    K_train = K[np.ix_(train_indices, train_indices)]
    K_test = K[np.ix_(test_indices, test_indices)]
    K_test_train = K[np.ix_(test_indices, train_indices)]
    K_sampled_train = K_train[np.ix_(sampled_order_train, sampled_order_train)]

    y_train = y[train_indices]
    y_test = y[test_indices]
    y_sampled_train = y_train[sampled_order_train]

    n_train = K_train.shape[0]
    n_test = K_test.shape[0]

    if stop_t is None:
        stop_t = n_train

    learning_curve_train = np.zeros(stop_t)
    learning_curve_test = np.zeros(stop_t)

    for t in range(stop_t):
        K_sampled_train_t = K_sampled_train[0:t+1, 0:t+1]
        gaussian_krc.fit(X=K_sampled_train_t, y=y_sampled_train[:t+1])

        # Predict for train set
        K_xn_train = K_train[np.ix_(
            np.arange(n_train), sampled_order_train[:t+1])]
        y_train_ = gaussian_krc.predict(K_xn_train)
        learning_curve_train[t] = accuracy_score(y_train, y_train_)

        # Then test set
        K_xn_test = K_test_train[np.ix_(
            np.arange(n_test), sampled_order_train[:t+1])]
        y_test_ = gaussian_krc.predict(K_xn_test)
        learning_curve_test[t] = accuracy_score(y_test, y_test_)

    return learning_curve_train, learning_curve_test


def sample_mc_learning_curves_train_test(K, y, train_indices, test_indices, tau, num_trajectories=5):
    """Sample learning curves when using a stochastic herding algorithm

    :param sampling_algorithm: (src.alg herding algorithm) instance of kernel herding algorithm
    """
    mc = alg.MCSampling(K[np.ix_(train_indices, train_indices)])

    learning_curves_mc_train = []
    learning_curves_mc_test = []
    for i in range(num_trajectories):
        mc.run_mc()
        lc_train, lc_test = calculate_learning_curves_train_test(K=K, y=y,
                                                                 train_indices=train_indices,
                                                                 test_indices=test_indices,
                                                                 sampled_order_train=mc.sampled_order,
                                                                 tau=tau)
        learning_curves_mc_train.append(lc_train)
        learning_curves_mc_test.append(lc_test)

    learning_curves_mc_train = np.array(learning_curves_mc_train)
    learning_curves_mc_test = np.array(learning_curves_mc_test)
    return learning_curves_mc_train, learning_curves_mc_test


def sample_learning_curves_for_random_algorithm(sampling_algorithm, K, y, train_indices, test_indices, tau, num_trajectories=5):
    """Sample learning curves (train, test) when using a stochastic herding algorithm"""
    algo = sampling_algorithm(K[np.ix_(train_indices, train_indices)])

    learning_curves_train = []
    learning_curves_test = []
    for i in range(num_trajectories):
        algo.run()
        lc_train, lc_test = calculate_learning_curves_train_test(K=K, y=y,
                                                                 train_indices=train_indices,
                                                                 test_indices=test_indices,
                                                                 sampled_order_train=algo.sampled_order,
                                                                 tau=tau)
        learning_curves_train.append(lc_train)
        learning_curves_test.append(lc_test)

    learning_curves_train = np.array(learning_curves_train)
    learning_curves_test = np.array(learning_curves_test)
    return learning_curves_train, learning_curves_test


def get_train_test_val_split(X, y, val_ratio=0.2):
    """Split dataset (X, y) into validation and test+train split

    This is mainly a utility function to help with cross validation"""
    n_full = X.shape[0]
    val_idx = int(np.floor(n_full * val_ratio))
    X_val, y_val = X[:val_idx], y[:val_idx]
    X_tr_te, y_tr_te = X[val_idx:], y[val_idx:]
    return X_tr_te, y_tr_te, X_val, y_val


def create_cross_validated_object(X_val, y_val):
    """Get the optimal cross validation parameters using validation dataset

    :param X_val: (np.ndarray, (n, d)) validation design matrix
    :param y_val: (np.ndarray, (n,)) validation array

    :return gkrc_cs: (sklearn.model) cross validation GaussianKRRClassification object
    :return tau_opt: (float) optimal tau for KRRC
    :return s2_opt: (float) optimal s2 KRRC"""
    gkrc_cv, tau_opt, s2_opt = pick_optimal_params_using_cv(
        X_val, y_val)

    return gkrc_cv, tau_opt, s2_opt


def run_learning_curve_experiment_k_fold(X, y, dataset_name, val_ratio=0.2, k_folds=5, num_trajectories=5):
    """Writes learning curves (using k-folds in order to get more stable results) to file in .npy format

    Run k-fold cross validation on the dataset for all of the algorithms, passed
    through deterministic_config and random_config. The dictionaries are built as follows

    :param X: (np.ndarray, (n, d)) design matrix
    :param y: (np.ndarray, (n,)) output
    :param dataset_name: (str) string containing the dataset name
    :param val_ratio: (float) ratio of dataset to use to get optimal parameters
    :param k_fold: (int) number of folds to use
    :param num_trajectories: (int) number of trajectories to run mc algorithm for
    """

    save_dir = 'learning_curves_k_fold-{}'.format(dataset_name)
    save_dir = Path(data_experiments_dir) / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # First split data up into train+test and validation
    # [train+test|validation]
    # Then get the optimal hyper-parameters
    X_tr_te, y_tr_te, X_val, y_val = get_train_test_val_split(X, y, val_ratio)
    gkrc_cs, tau_opt, s2_opt = create_cross_validated_object(
        X_val, y_val)

    # Output the rest of the data
    K, K_mmd = create_krr_mmd_kernel_matrices(X_tr_te, s2_opt)
    n = K.shape[0]
    block_size = int(np.floor(n / k_folds))

    # We then split up the train and test indices using k_folds
    train_indices_list, test_indices_list = k_fold_train_test_indices(
        n, k_folds)

    # Deterministic algorithms
    # dims: fold, learning_curve
    learning_curves_levscore_train_k_folds = np.zeros(
        (k_folds, block_size * (k_folds - 1)))
    learning_curves_levscore_test_k_folds = np.zeros(
        (k_folds, block_size * (k_folds - 1)))
    learning_curves_fw_train_k_folds = np.zeros(
        (k_folds, block_size * (k_folds - 1)))
    learning_curves_fw_test_k_folds = np.zeros(
        (k_folds, block_size * (k_folds - 1)))
    # Random algorithms
    # dims: fold, sampled_trajectory, learning_curve
    learning_curves_mc_train_k_folds = np.zeros(
        (k_folds, num_trajectories, block_size * (k_folds - 1)))
    learning_curves_mc_test_k_folds = np.zeros(
        (k_folds, num_trajectories, block_size * (k_folds - 1)))

    for fold, (train_indices, test_indices) in enumerate(zip(train_indices_list, test_indices_list)):
        print('Running fold: {}'.format(fold))
        learning_curves_mc_train, learning_curves_mc_test = sample_mc_learning_curves_train_test(
            K, y_tr_te, train_indices, test_indices, tau_opt, num_trajectories=num_trajectories)
        learning_curves_mc_train_k_folds[fold,
                                         :, :] = learning_curves_mc_train.copy()
        learning_curves_mc_test_k_folds[fold,
                                        :, :] = learning_curves_mc_test.copy()

        K_train = K[np.ix_(train_indices, train_indices)]
        fw = alg.FrankWolfe(K_train)
        fw.run_frank_wolfe()
        learning_curve_fw_train, learning_curve_fw_test = calculate_learning_curves_train_test(K, y_tr_te,
                                                                                               train_indices,
                                                                                               test_indices,
                                                                                               fw.sampled_order,
                                                                                               tau_opt)
        learning_curves_fw_train_k_folds[fold] = learning_curve_fw_train.copy()
        learning_curves_fw_test_k_folds[fold] = learning_curve_fw_test.copy()

        levscore = alg.LeverageScoreSampling(K_train, tau_opt)
        levscore.run()
        learning_curve_levscore_train, learning_curve_levscore_test = calculate_learning_curves_train_test(K, y_tr_te,
                                                                                                           train_indices,
                                                                                                           test_indices,
                                                                                                           levscore.sampled_order,
                                                                                                           tau_opt)
        learning_curves_levscore_train_k_folds[fold] = learning_curve_levscore_train.copy(
        )
        learning_curves_levscore_test_k_folds[fold] = learning_curve_levscore_test.copy(
        )

    # Save all of the learning curves
    np.save(save_dir / 'learning_curves_fw_train_k_folds',
            learning_curves_fw_train_k_folds)
    np.save(save_dir / 'learning_curves_fw_test_k_folds',
            learning_curves_fw_test_k_folds)
    np.save(save_dir / 'learning_curves_levscore_train_k_folds',
            learning_curves_levscore_train_k_folds)
    np.save(save_dir / 'learning_curves_levscore_test_k_folds',
            learning_curves_levscore_test_k_folds)
    np.save(save_dir / 'learning_curves_mc_train_k_folds',
            learning_curves_mc_train_k_folds)
    np.save(save_dir / 'learning_curves_mc_test_k_folds',
            learning_curves_mc_test_k_folds)

    # Save json file of interesting information for this particular run
    euclidean_dist_q05 = kernel_quantile_heuristic(X, q=0.05)
    euclidean_dist_q95 = kernel_quantile_heuristic(X, q=0.95)

    param_config = {
        'n': X.shape[0],
        'n_tr_te': X_tr_te.shape[0],
        'd': X.shape[1],
        'k_folds': k_folds,
        'test_fold_size': block_size,
        'tau_opt_KRR': tau_opt,
        's2_opt_KRR': s2_opt,
        'euclidean_dist_q05': euclidean_dist_q05,
        'euclidean_dist_q95': euclidean_dist_q95,
        'time_created': str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    }

    with open(save_dir / 'experiment_config.json', 'w') as json_file:
        json.dump(param_config, json_file)


def save_learning_curve_k_fold_plot(experiment_dir_name, traces=True, plot_type='plot'):
    experiment_dir = Path(data_experiments_dir) / experiment_dir_name

    # Load mc learning curves
    learning_curves_mc_train = np.load(
        experiment_dir / 'learning_curves_mc_train_k_folds.npy')
    learning_curves_mc_test = np.load(
        experiment_dir / 'learning_curves_mc_test_k_folds.npy')
    learning_curves_fw_train = np.load(
        experiment_dir / 'learning_curves_fw_train_k_folds.npy')
    learning_curves_fw_test = np.load(
        experiment_dir / 'learning_curves_fw_test_k_folds.npy')
    learning_curves_levscore_train = np.load(
        experiment_dir / 'learning_curves_levscore_train_k_folds.npy')
    learning_curves_levscore_test = np.load(
        experiment_dir / 'learning_curves_levscore_test_k_folds.npy')

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    if traces:
        viz.plot_learning_curves_traces_all_algorithms_k_fold(
            learning_curves_mc_test, learning_curves_fw_test, learning_curves_levscore_test, fig=fig, ax=ax[0], plot_type=plot_type)
        viz.plot_learning_curves_traces_all_algorithms_k_fold(
            learning_curves_mc_train, learning_curves_fw_train, learning_curves_levscore_train, fig=fig, ax=ax[1], plot_type=plot_type)
    else:
        viz.plot_learning_curves_all_algorithms_k_fold(
            learning_curves_mc_test, learning_curves_fw_test, learning_curves_levscore_test, fig=fig, ax=ax[0], plot_type=plot_type)
        viz.plot_learning_curves_all_algorithms_k_fold(
            learning_curves_mc_train, learning_curves_fw_train, learning_curves_levscore_train, fig=fig, ax=ax[1], plot_type=plot_type)

    # Style plot: accuracy is always between 0 and 1
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('t')
    ax[0].set_ylim([0.0, 1.0])
    ax[0].set_title('Test set')

    ax[1].set_xlabel('t')
    ax[1].set_ylim([0.0, 1.0])
    ax[1].set_title('Train set')
    ax[1].legend()

    fig.savefig(Path(img_dir) / experiment_dir_name)


###############################################
# sklearn type custom kernel ridge regression #
###############################################


class GaussianKRRClassification(BaseEstimator, ClassifierMixin):
    """We follow the Carlo Ciliberto's

    A Consistent Regularization Approachfor Structured Prediction

    for implementing classification using KRR together with a minimisation problem.

    This means that KRR is of the form min sum_i^n (f(x_i) - y_i)**2 + tau\|f\|_H**2
    leading to alpha = (K + tau * I)^-1 Y.

    We assume that Y is of the form (np.ndarray, n) and that
    if we have C classes they are encoded from 0:C-1."""

    def __init__(self, tau=None, s2=None, precompute_K=False):
        self.tau = tau
        self.s2 = s2
        self.precompute_K = precompute_K

    def fit(self, X, y):
        # Most of computation takes place at predict time
        if self.precompute_K:
            assert X.shape[0] == X.shape[1], "K needs to be square"
            self.K = X
        else:
            assert X.ndim == 2
            assert y.shape[0] == X.shape[0]
            self.X = X
            self.K = gaussian_kernel_matrix(X, s2=self.s2)

        self.n = self.K.shape[0]

        # We need to cover for cases when the train output miss certain classes
        # consider https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
        self.y = np.atleast_1d(y.squeeze())
        self.train_classes = np.unique(self.y).astype(int)
        self.train_classes_enumerated = np.arange(
            self.train_classes.shape[0]).astype(int)
        self.from_class_to_enum = np.vstack(
            (self.train_classes, self.train_classes_enumerated))
        self.from_enum_to_class = np.vstack(
            (self.train_classes_enumerated, self.train_classes))

        self.a = (self.K + self.tau * np.eye(self.n))

        return self

    def _map_from_classes_to_enumerated(self, y):
        _y = np.atleast_1d(y.copy().astype(int).squeeze())
        # Classes not in train set fail automatically
        _y[~np.isin(_y, self.train_classes)] = -1
        mask = np.in1d(_y, self.from_class_to_enum[0, :])
        _y[mask] = self.from_class_to_enum[1, np.searchsorted(
            self.from_class_to_enum[0, :], _y[mask])]
        return _y

    def _map_from_enumerated_to_classes(self, y_enumerated):
        _y_enumerated = np.atleast_1d(
            y_enumerated.copy().astype(int).squeeze())
        mask = np.in1d(_y_enumerated, self.from_enum_to_class[0, :])
        _y_enumerated[mask] = self.from_enum_to_class[1, np.searchsorted(
            self.from_enum_to_class[0, :], _y_enumerated[mask])]
        return _y_enumerated

    def _build_D_C(self):
        """Helper function to build D_C matrix from y"""
        _y_enum = self._map_from_classes_to_enumerated(self.y)
        return (self.train_classes_enumerated.reshape(-1, 1) != _y_enum.reshape(1, -1)).astype(int)

    def predict(self, X):
        """
        Explanation: the prediction works as follows,
        We first solve a sub-problem of fitting alpha
        where alpha_pred is [n_train, n_pred] and is the solution to the equation
        a * alpha_pred = K_{n_train, n_pred}
        and then calculating the loss matrix D_C where each row of D_C
        is D_i where D_ij = Loss(i, y_j) = I(i != y_j) and j is one of
        the indices of the train set.

        :param X: (np.ndarray, (n_pred, n)) cross-kernel matrix between X_pred and X_tr (or X_pred)

        :return y_pred: (np.ndarray, (n_pred,))
        """
        if self.precompute_K:
            # in this case X is the cross-kernel-matrix of (X_train, X_pred)
            K_nn_pred = X.T
        else:
            K_nn_pred = gaussian_kernel_matrix(self.X, X, s2=self.s2)
        D_C = self._build_D_C()
        assert D_C.shape == (self.train_classes_enumerated.shape[0], self.n), "D_C should have dimensions ({}, {}), has dimension {}".format(
            self.train_classes_enumerated.shape[0], self.K.shape[0], D_C.shape)

        alpha_pred = linalg.solve(a=self.a, b=K_nn_pred, assume_a='pos')
        assert alpha_pred.shape == (self.n, X.shape[0])

        y_pred_enum = (D_C @ alpha_pred).argmin(
            axis=0)  # Want this to have ndims == 1
        y_pred_enum = np.atleast_1d(y_pred_enum.squeeze())
        y_pred = self._map_from_enumerated_to_classes(y_pred_enum)
        return y_pred

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def get_params(self, deep=True):
        return {"tau": self.tau, "s2": self.s2}

    def set_params(self, tau, s2):
        self.tau = tau
        self.s2 = s2
        return self