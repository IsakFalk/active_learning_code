import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error


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


def visualise_mmd_curve(mmd_curve, loglog=True):
    """Visualise the trace of the mmd score as a function of the size

    Given a curve which has mmd_curve[t] = MMD(P, Q_t), visualise
    both the normal plot and the log plot.

    :param mmd_curve: (1 x n, np.ndarray) mmd scores
    :param loglog: (bool) if true, plot loglog, else semilogy

    :return fig, ax:"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # first we do normal plot
    ax[0].plot(mmd_curve, linestyle='-')
    ax[0].set_xlabel(r't')
    ax[0].set_ylabel(r'$MMD(\hat{P}, \hat{Q}_t)$')
    ax[0].set_ylim([0.0, 1.0])
    ax[0].set_title('MMD vs t')

    # Then we do semilogyx to get the actual relationship
    # NB: If we have relationships of kind MMD = O(t**-a)
    # we will have relationships of the kind
    # log(y) = log(C) - a log(t)
    # If we have relationships like MMD = A exp(-O(t))
    # we will have relationships of the kind
    # log(y) = log(A) - C * t
    # Choose the plot that makes sense given the method visualising
    t = np.arange(0, n)
    if loglog:
        ax[1].loglog(t[0:-1], mmd_curve[0:-1], linestyle='-', marker='^')
        ax[1].set_title('MMD vs t, loglog-plot')
    else:
        ax[1].semilogy(t[0:-1], mmd_curve[0:-1], linestyle='-', marker='^')
        ax[1].set_title('MMD vs t, logy-plot')

    return fig, ax


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


###############################################
# sklearn type custom kernel ridge regression #
###############################################


class GaussianKernelRidgeRegression(BaseEstimator, RegressorMixin):
    """This is of the form min sum_i^n (f(x_i) - y_i)**2 + tau\|f\|_H**2
       leading to alpha = (K + tau * I)^-1 Y"""

    def __init__(self, tau, s2):
        self.tau = tau
        self.s2 = s2

    def fit(self, X, y):
        assert X.ndim == 2
        assert y.ndim == 2
        assert y.shape[0] == X.shape[0]
        assert y.shape[1] == 1
        self.n, self.d = X.shape
        self.X = X
        self.K = gaussian_kernel_matrix(X, s2=self.s2)
        a = (self.K + self.tau * np.eye(self.n))
        self.alpha = linalg.solve(a=a, b=y, assume_a='pos')

    def predict(self, X):
        K_xn = gaussian_kernel_matrix(X, self.X, s2=self.s2)
        return K_xn @ self.alpha

    def score(self, X, y=None):
        return -mean_squared_error(self.predict(X), y)

    def get_params(self, deep=True):
        return {"tau": self.tau, "s2": self.s2}

    def set_params(self, tau, s2):
        self.tau = tau
        self.s2 = s2
        return self
