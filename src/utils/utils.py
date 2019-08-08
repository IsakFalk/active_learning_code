import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

import src.algorithms as alg
import src.visualization.visualize as viz
from src.PARAMATERS import data_experiments_dir, img_dir


def subsample_dataset(X, y, n_subsample=700):
    n, d = X.shape
    subsample_indices = np.random.permutation(n)[:n_subsample]
    return X[subsample_indices], y[subsample_indices]


def gaussian_kernel_matrix(X, Y=None, s2=1.0):
    """
    X: [n, d] matrix, where n is the number of samples, d is the dimension
    s2: the standard deviation parameter, K(x, y) = np.exp(-np.norm(x - y)**2/(2*s2))
    """
    # Get the matrix K_XY
    if type(Y) == np.ndarray or type(Y) == np.ndarray:
        pairwise_sq_dists = cdist(X, Y, 'sqeuclidean')
        K = np.exp(-0.5 * pairwise_sq_dists / s2)
    # Get the matrix K_XX
    else:
        pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
        K = np.exp(-0.5 * pairwise_sq_dists / s2)
    return K


def mmd2(K_xx, K_yy, K_xy, w_x=None, w_y=None):
    """
    Calculate the mmd ** 2 between two empirical samples X and Y

    Allow for weighted sums, such that
    \mu_x = \sum_i w_xi K(x_i, )
    and
    \mu_y = \sum_j w_yj K(y_j, )

    :param K_xx: (np.ndarray, (n, n)) kernel matrix constructed from X
    :param K_yy: (np.ndarray, (m, m)) kernel matrix constructed from Y
    :param K_xy: (np.ndarray, (n, m)) kernel matrix constructed from X (rows) and Y (cols)
    :param w_x: (np.ndarray, (n, 1)) weights of datapoints in X, is a distribution
    :param w_y: (np.ndarray, (m, 1)) weights of datapoints in Y, is a distribution
    """
    n, m = K_xy.shape

    assert n == K_xx.shape[0], "Shapes must conform between K_xx and K_xy, K_xx.shape == {}, K_xy.shape == {}".format(
        K_xx.shape, K_xy.shape)
    assert m == K_yy.shape[0], "Shapes must conform between K_yy and K_xy, K_yy.shape == {}, K_xy.shape == {}".format(
        K_yy.shape, K_xy.shape)

    if isinstance(w_x, np.ndarray) and isinstance(w_y, np.ndarray):
        assert np.isclose(w_x.sum(), 1) and np.isclose(
            w_y.sum(), 1), "w_x and w_y must sum to 1"
        assert w_x.shape == (n, 1) and w_y.shape == (
            m, 1), "w_x and w_y must conform to K_xx and K_yy, have w_x.shape == {}, w_y.shape == {} and K_xx.shape == {}, K_yy == {}".format(w_x.shape, w_y.shape, K_xx.shape, K_yy.shape)
        assert (w_x >= 0).all(), "All entries of w_x should be greater than zero"
        assert (w_y >= 0).all(), "All entries of w_y should be greater than zero"
        mmd2 = w_x.T @ K_xx @ w_x - 2 * w_x.T @ K_xy @ w_y + w_y.T @ K_yy @ w_y
    else:
        mmd2 = (K_xx.sum() / (n**2)) + (K_yy.sum() / (m**2)) - \
            2 * (K_xy.sum() / (m*n))

    # Had problem with negative values on order of machine epsilon
    mmd2 += 2 * np.finfo(float).eps
    assert mmd2 > 0.0, "mmd2 should be non-negative, is {}".format(mmd2)

    return mmd2


def find_kernel_diameter(K):
    """Find the diameter of the polytope in RKHS"""
    n = K.shape[0]
    best_max = 0
    best_indices = [0, 0]
    current_max = 0
    current_indices = [0, 0]
    for i in range(0, n):
        for j in range(0, i+1):
            current_max = K[i, i] + K[j, j] - 2*K[i, j]
            current_indices = [i, j]
            if current_max > best_max:
                best_max = current_max
                best_indices = current_indices

    return best_max, best_indices


def calculate_mmd_curve(K, sampled_order, W=None):
    """Get the whole mmd scores for the sampling strategy that produce X_sampled

    Get the mmd scores for all the empirical sampling distribution produced by
    X_sampled[0:t] up to time t, where mmd_array[t] = MMD(X, X_sampled[0:t])_K

    :param K: (np.ndarray, (n, n)) original kernel matrix
    :param sampled_order: (np.ndarray, (stop_t,)) order of index sampled by algorithm
    :param W: (np.ndarray, (n, n)) the weight matrix (for frank wolfe)

    :return mmd_curve: (np.ndarray, (stop_t,)) the curve of mmd scores
    """
    n = K.shape[0]
    stop_t = sampled_order.shape[0]

    mmd_curve = np.zeros(stop_t)
    K_xx = K
    K_xtxt = K[np.ix_(sampled_order, sampled_order)]
    # If we are in weighted case we use this
    w_x = np.ones(n).reshape(-1, 1) / n
    for t in range(stop_t):
        K_yy = K_xtxt[:t+1, :t+1]
        K_xy = K[np.ix_(np.arange(n), sampled_order[:t+1])]
        if W is None:
            mmd_curve[t] = mmd2(K_xx, K_yy, K_xy) ** 0.5
        else:
            w_y = W[t, :t+1].reshape(t+1, 1)
            mmd_curve[t] = mmd2(K_xx, K_yy, K_xy, w_x=w_x,
                                w_y=w_y) ** 0.5

    return mmd_curve


def create_f(x_is, alpha_is, s2=1.0, normalise=True):
    """
    create a function from RKHS, f(x) = \sum_i^n alpha_i * K(x_i, x) = \alpha^T K(x_i, :)
    :param x_is: the points of K(x_i, \cdot)
    :param alpha_is: the coefficients (1 x n)
    :param s2: gaussian kernel variance
    :param normaliser: if f should have unit norm by normalising

    :return f: a function takin as input a 2xn np.ndarray and outputs an 1xn np.ndarray
    """
    normaliser = 1.0
    if normalise:
        # In our case we have that ||f||_H^2 can be written in terms of
        # the points and coefficients as alpha.T K alpha
        normaliser = np.sqrt(alpha_is.T @ gaussian_kernel_matrix(
            x_is, x_is, s2=s2) @ alpha_is).squeeze()

    def f(x):
        return gaussian_kernel_matrix(x, x_is, s2=s2) @ alpha_is / normaliser
    return f


def get_i_eigen_function(i, X, K, times_bigger_than_machine_epsilon=1.0):
    """
    We get the i'th function as defined by the badly conditioned criteria, we start from below
    f_i(x) = \sum_{j=1}^n K(x_j, x) u_ij
    """
    la, v = linalg.eigh(K)  # eigh uses the knowledge that K is symmetric p.s.d
    cutoff = times_bigger_than_machine_epsilon * np.finfo(float).eps
    # Get the index of the first element greater than cutoff
    cutoff_i = np.arange(la.shape[0])[la > cutoff][0]
    i_ = max(cutoff_i, i)
    f = create_f(x_is=X, alpha_is=v[i_, :])
    return f


def get_extremum(f,
                 dx=0.05, dy=0.05,
                 ylims=[-2, 2], xlims=[-2, 2]):
    """
    Get extremum 2d function f which takes a vector 2x1 as input

    The functionality is taken from https://matplotlib.org/examples/pylab_examples/pcolor_demo.html

    :param f: (function object) function that teaks a 1x2 np.array and outputs a real number
    :param dx: step-size of x-axis
    :param dy: step-size of y-axis
    :param ylims: (list) list of min and max of y-axis
    :param xlims: (list) list of min and max of x-axis"""
    # generate 2 2d grids for the x & y bounds
    ymin, ymax = ylims
    xmin, xmax = xlims
    y, x = np.mgrid[slice(ymin, ymax + dy, dy),
                    slice(xmin, xmax + dx, dx)]

    # create f(x, y)
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            xy = np.array([x[i, j], y[i, j]]).reshape(1, -1)
            z[i, j] = f(xy)

    return np.max(z), np.min(z)


def kernel_quantile_heuristic(X, q=0.5):
    """Calculates the optimal s2 given the dataset X

    Using the quantile heurstic we calculate the best s2.

    :return quantile_heuristic_s2: (float) s2 picked according to quantile heuristic"""
    n, d = X.shape
    pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
    pairwise_sq_dists_unbiased = pairwise_sq_dists[np.triu_indices(n, k=1)]
    quantile_heuristic_s2 = np.quantile(pairwise_sq_dists_unbiased, q=q)
    return quantile_heuristic_s2


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


def train_test_indices(n, train_ratio=0.7):
    """Create train and test indices

    Create a new random index for test and train set
    using the train_ratio as the proportion of train to
    test data. Note that this function removes remainder
    from indices, the blocks are all of same size.

    :param n: (int) size of original dataset
    :param train_ratio: (float) ratio of points that should be training data

    :return train_indices: (np.ndarray) train indices
    :return test_indices: (np.ndarray) test indices"""
    train_split_index = int(train_ratio * n)
    shuffled_indices = np.random.permutation(n)
    train_indices = shuffled_indices[:train_split_index]
    test_indices = shuffled_indices[train_split_index:]
    return train_indices, test_indices


def create_krr_mmd_kernel_matrices(X, s2_opt):
    """s2_opt is the optimal s2 hyperparameter for KRR"""
    K = gaussian_kernel_matrix(X, s2=s2_opt)
    K_mmd = gaussian_kernel_matrix(X, s2=s2_opt/2.0)
    return K, K_mmd


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


def run_learning_curve_experiment(X, y, dataset_name, train_ratio=0.7):
    """Writes learning curves to file in .npy format

    Will want to change what we call in this as more algorithms are added"""
    gkr_cv, tau_opt, s2_opt = pick_optimal_params_using_cv(X, y)
    K, K_mmd = create_krr_mmd_kernel_matrices(X, s2_opt)
    n = K.shape[0]
    train_indices, test_indices = train_test_indices(
        n, train_ratio=train_ratio)

    learning_curves_mc_train, learning_curves_mc_test = sample_mc_learning_curves_train_test(
        K, y, train_indices, test_indices, tau_opt, num_trajectories=10)

    K_train = K[np.ix_(train_indices, train_indices)]
    fw = alg.FrankWolfe(K_train)
    fw.run_frank_wolfe()
    learning_curve_fw_train, learning_curve_fw_test = calculate_learning_curves_train_test(K, y,
                                                                                           train_indices,
                                                                                           test_indices,
                                                                                           fw.sampled_order,
                                                                                           tau_opt)

    save_dir = 'learning_curves-{}'.format(dataset_name)
    save_dir = Path(data_experiments_dir) / save_dir
    print(save_dir)
    save_dir.mkdir(parents=True, exist_ok=False)

    # Save all of the learning curves
    np.save(save_dir / 'learning_curve_fw_train', learning_curve_fw_train)
    np.save(save_dir / 'learning_curve_fw_test', learning_curve_fw_test)
    np.save(save_dir / 'learning_curves_mc_train', learning_curves_mc_train)
    np.save(save_dir / 'learning_curves_mc_test', learning_curves_mc_test)

    # Save json file of interesting information for this particular run
    euclidean_dist_q05 = kernel_quantile_heuristic(X, q=0.05)
    euclidean_dist_q95 = kernel_quantile_heuristic(X, q=0.95)

    param_config = {
        'n': X.shape[0],
        'd': X.shape[1],
        'tau_opt_KRR': tau_opt,
        's2_opt_KRR': s2_opt,
        'train_ratio': train_ratio,
        'euclidean_dist_q05': euclidean_dist_q05,
        'euclidean_dist_q95': euclidean_dist_q95,
        'time_created': str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    }

    with open(save_dir / 'experiment_config.json', 'w') as json_file:
        json.dump(param_config, json_file)


def k_fold_train_test_indices(n, k):
    block_size = int(np.floor(n / k))
    end_n = block_size * k
    train_indices_list = []
    test_indices_list = []
    for fold in range(k):
        if fold == 0:
            train_array = np.arange(block_size, end_n)
            test_array = np.arange(0,  block_size)
        elif fold == k-1:
            train_array = np.arange(0, fold * block_size)
            test_array = np.arange(fold * block_size, end_n)
        else:
            train_array_1 = np.arange(0, fold * block_size)
            test_array = np.arange(fold * block_size, (fold+1) * block_size)
            train_array_2 = np.arange((fold + 1) * block_size, end_n)
            train_array = np.concatenate((train_array_1, train_array_2))
        train_indices_list.append(train_array)
        test_indices_list.append(test_array)
    return train_indices_list, test_indices_list


def run_learning_curve_experiment_k_fold(X, y, dataset_name, val_ratio=0.2, k_folds=5, num_trajectories=10):
    """Writes learning curves (using k-folds in order to get more stable results) to file in .npy format

    Will want to change what we call in this as more algorithms are added"""

    save_dir = 'learning_curves_k_fold-{}'.format(dataset_name)
    save_dir = Path(data_experiments_dir) / save_dir
    save_dir.mkdir(parents=True, exist_ok=False)

    # First split data up into train+test and validation
    # [train+test|validation]
    # Then get the optimal hyper-parameters
    n_full = X.shape[0]
    val_idx = int(np.floor(n_full * val_ratio))
    X_val, y_val = X[:val_idx], y[:val_idx]
    X_tr_te, y_tr_te = X[val_idx:], y[val_idx:]
    gkr_cv, tau_opt, s2_opt = pick_optimal_params_using_cv(X_val, y_val)
    # Output the rest of the data
    K, K_mmd = create_krr_mmd_kernel_matrices(X_tr_te, s2_opt)
    n = K.shape[0]
    block_size = int(np.floor(n / k_folds))

    # We then split up the train and test indices using k_folds
    train_indices_list, test_indices_list = k_fold_train_test_indices(
        n, k_folds)

    learning_curves_fw_train_k_folds = np.zeros(
        (k_folds, block_size * (k_folds - 1)))
    learning_curves_fw_test_k_folds = np.zeros(
        (k_folds, block_size * (k_folds - 1)))
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

    # Save all of the learning curves
    np.save(save_dir / 'learning_curves_fw_train_k_folds',
            learning_curves_fw_train_k_folds)
    np.save(save_dir / 'learning_curves_fw_test_k_folds',
            learning_curves_fw_test_k_folds)
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
        'n_val': n_full - n,
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


def save_learning_curve_plot(experiment_dir_name):
    experiment_dir = Path(data_experiments_dir) / experiment_dir_name

    # Load mc learning curves
    learning_curves_mc_train = np.load(
        experiment_dir / 'learning_curves_mc_train.npy')
    learning_curves_mc_test = np.load(
        experiment_dir / 'learning_curves_mc_test.npy')
    learning_curve_fw_train = np.load(
        experiment_dir / 'learning_curve_fw_train.npy')
    learning_curve_fw_test = np.load(
        experiment_dir / 'learning_curve_fw_test.npy')
    print(learning_curves_mc_test.shape)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    viz.plot_learning_curves_mc_vs_kh(
        learning_curves_mc_test, learning_curve_fw_test, plot_type='semilogy', fig=fig, ax=ax[0])
    viz.plot_learning_curves_mc_vs_kh(
        learning_curves_mc_train, learning_curve_fw_train, plot_type='semilogy', fig=fig, ax=ax[1])

    supervised_limit = learning_curve_fw_test[-1]
    ax[0].axhline(y=supervised_limit, color='black',
                  linestyle='--', label='Average loss on full dataset')
    supervised_limit = learning_curve_fw_train[-1]
    ax[1].axhline(y=supervised_limit, color='black',
                  linestyle='--', label='Average loss on full dataset')

    ax[0].set_title('Test loss')
    ax[1].set_title('Train loss')
    ax[1].legend()

    fig.savefig(Path(img_dir) / experiment_dir_name)


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

    ax[0].set_title('Test loss')
    ax[1].set_title('Train loss')
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


#######################################
# Create SVD sigma for various decays #
#######################################

def create_sigma(n, decay_func):
    S = np.arange(1, n+1).astype(float)
    S = decay_func(S)
    S = np.diag(S)
    return(S)


def create_power_decay_sigma(n, alpha):
    def power_decay_func(S):
        assert S.dtype == float
        return S ** (-alpha)
    S = create_sigma(n, power_decay_func)
    return S


def create_exp_decay_sigma(n, beta):
    def exp_decay_func(S):
        assert S.dtype == float
        return np.exp(-beta * S)
    S = create_sigma(n, exp_decay_func)
    return S


def create_tsvd_sigma(S, th):
    # S (n, n) matrix not array
    diag_S = np.diag(S).copy()
    diag_S[diag_S < th] = 0.0
    return_S = np.diag(diag_S)
    return return_S
