import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


def gaussian_kernel_matrix(X, Y=None, s2=1.0):
    """
    X: [n, d] matrix, where n is the number of samples, d is the dimension
    s2: the standard deviation parameter, K(x, y) = np.exp(-np.norm(x - y)**2/s2)
    """
    # Get the matrix K_XY
    if type(Y) == np.ndarray or type(Y) == np.ndarray:
        pairwise_sq_dists = cdist(X, Y, 'sqeuclidean')
        K = np.exp(-pairwise_sq_dists / s2)
    # Get the matrix K_XX
    else:
        pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
        K = np.exp(-pairwise_sq_dists / s2)
    return K


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


def get_extremum(f,
                 dx=0.05, dy=0.05,
                 ylims=[-2, 2], xlims=[-2, 2]):
    """Get extremum 2d function f which takes a vector 2x1 as input

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
