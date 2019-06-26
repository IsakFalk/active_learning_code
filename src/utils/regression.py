from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

import src.visualization.visualize as viz
from src.PARAMATERS import data_experiments_dir, img_dir
from src.utils.utils import gaussian_kernel_matrix, kernel_quantile_heuristic


def pick_optimal_params_using_cv(X, y, cv=5, iid=False):
    """Using cross validation on dataset, pick best parameters

    Using the gaussian kernel ridge regression, pick the best tau and s2
    using cross validation on the data. Internally use the 0.05 and 0.95
    quantile as base low and high values to linspace over.

    :param X: (np.ndarray, (n, d)) design matrix
    :param y: (np.ndarray, (n, 1)) output array
    :param cv: (int) number of folds in cross-validation
    :param iid: (bool) if we assume iid or not (False lead to normal k-fold cv)

    :return gkr_cv: (GaussianKernelRidgeRegression) optimised gkr instance through sklearn GridSearchCV
    :rerurn tau: (float) optimal tau for GKRR
    :return s2: (float) optimal s2 for GKRR"""
    gaussian_kr = GaussianKernelRidgeRegression()

    q10_sq_dist = kernel_quantile_heuristic(X, q=0.05)
    q90_sq_dist = kernel_quantile_heuristic(X, q=0.95)

    param_grid = dict(
        tau=np.logspace(-3, 1, 5),
        s2=np.linspace(q10_sq_dist * 0.1, q90_sq_dist * 10, 5)
    )

    gkr_cv = GridSearchCV(estimator=gaussian_kr,
                          param_grid=param_grid,
                          scoring='neg_mean_squared_error',
                          cv=cv, iid=iid)
    gkr_cv.fit(X, y)
    tau = gkr_cv.best_params_['tau']
    s2 = gkr_cv.best_params_['s2']
    return gkr_cv, tau, s2


def calculate_learning_curve(K, y, sampled_order, tau, stop_t=None):
    """Calculate learning curves from running herding algorithm

    Using the sampled order from the sampled_order indexing array
    calculate the learning curve on the train set using GKRR. Note that we
    pass K instead of calculating it on the fly, that's why we don't use
    s2 explicitly, it's already used in calculating K.

    :param K: (np.ndarray, (n, n)) full kernel matrix from dataset
    :param y: (np.ndarray, (n, 1)) output array
    :param sampled_order: (np.ndarray, (n,)) order of the sampled indices
    :param tau: (float) regularisation parameter used in GKRR
    :param stop_t: (int) final step of calculations

    :return learning_curve: (np.ndarray, (stop_t,)) array of mse for size of train set
    """
    gaussian_kr = GaussianKernelRidgeRegression(
        tau=tau, s2=None, precompute_K=True)

    n = K.shape[0]
    if stop_t is None:
        stop_t = n

    all_indices = np.arange(n)
    K_sampled = K[np.ix_(sampled_order, sampled_order)]

    learning_curve = np.zeros(stop_t)
    for t in range(stop_t):
        K_sampled_t = K_sampled[0:t+1, 0:t+1]
        gaussian_kr.fit(X=K_sampled_t, y=y[sampled_order[:t+1]])
        # NB: the sampled_order index the train set
        # and all indices the set we are predicting
        K_xn = K[np.ix_(all_indices, sampled_order[:t+1])]
        y_ = gaussian_kr.predict(K_xn)
        learning_curve[t] = mean_squared_error(y, y_)

    return learning_curve


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
    gaussian_kr = GaussianKernelRidgeRegression(
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
        gaussian_kr.fit(X=K_sampled_train_t, y=y_sampled_train[:t+1])

        # Predict for train set
        K_xn_train = K_train[np.ix_(
            np.arange(n_train), sampled_order_train[:t+1])]
        y_train_ = gaussian_kr.predict(K_xn_train)
        learning_curve_train[t] = mean_squared_error(y_train, y_train_)

        # Then test set
        K_xn_test = K_test_train[np.ix_(
            np.arange(n_test), sampled_order_train[:t+1])]
        y_test_ = gaussian_kr.predict(K_xn_test)
        learning_curve_test[t] = mean_squared_error(y_test, y_test_)

    return learning_curve_train, learning_curve_test


def save_learning_curve_k_fold_plot(experiment_dir_name):
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

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    viz.plot_learning_curves_mc_vs_kh_k_fold(
        learning_curves_mc_test, learning_curves_fw_test, plot_type='semilogy', fig=fig, ax=ax[0])
    viz.plot_learning_curves_mc_vs_kh_k_fold(
        learning_curves_mc_train, learning_curves_fw_train, plot_type='semilogy', fig=fig, ax=ax[1])

    ax[0].set_title('Test set')
    ax[0].set_ylabel('MSE')
    ax[0].set_xlabel('t')
    ax[1].set_title('Train set')
    ax[1].set_xlabel('t')
    ax[1].legend()

    fig.savefig(Path(img_dir) / experiment_dir_name)


###############################################
# sklearn type custom kernel ridge regression #
###############################################


class GaussianKernelRidgeRegression(BaseEstimator, RegressorMixin):
    """This is of the form min sum_i^n (f(x_i) - y_i)**2 + tau\|f\|_H**2
       leading to alpha = (K + tau * I)^-1 Y"""

    def __init__(self, tau=None, s2=None, precompute_K=False):
        self.tau = tau
        self.s2 = s2
        self.precompute_K = precompute_K

    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        else:
            assert y.shape[1] == 1, "y needs to be univariate output"

        self.y = y
        self.n = X.shape[0]

        # Branch depending on if we reuse K or calculate it X
        if self.precompute_K:
            self.K = X
            assert self.K.shape == (X.shape[0], X.shape[0])
        else:
            assert X.ndim == 2
            assert y.shape[0] == X.shape[0]

            self.X = X
            self.K = gaussian_kernel_matrix(X, s2=self.s2)
        a = (self.K + self.tau * np.eye(self.n))
        self.alpha = linalg.solve(a=a, b=y, assume_a='pos')

    def predict(self, X):
        if self.precompute_K:
            # in this case X is the cross-kernel-matrix
            K_xn = X
        else:
            K_xn = gaussian_kernel_matrix(X, self.X, s2=self.s2)
        y_pred = K_xn @ self.alpha
        return y_pred

    def score(self, X, y):
        return -mean_squared_error(self.predict(X), y)

    def get_params(self, deep=True):
        return {"tau": self.tau, "s2": self.s2}

    def set_params(self, tau, s2):
        self.tau = tau
        self.s2 = s2
        return self
