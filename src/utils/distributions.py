import numpy as np
from sklearn.datasets import make_blobs


class MoG():
    def __init__(self, d=None, mus=None, sigmas=None, p=None):
        """
        param d: (int) dimension of gaussian components
        param mus: (np.array, (n_components, d)) array of means for gaussian components
        param sigmas: (np.array,  (n_components, d, d)) array of covariance matrices for gaussian components
        param p: (np.array, (n_components,)) probability array for categorical distribution over gaussians components
        """
        # define everything or fall back to default case
        if d is None or mus is None or sigmas is None or p is None:
            self.mus = np.array([[0], [1]])
            self.sigmas = np.array([[[1]], [[1]]])
            self.p = np.array([0.5, 0.5])
            self.d = self.mus.shape[1]
        else:
            assert d == mus.shape[1] and d == sigmas.shape[-1]
            self.mus = mus
            self.sigmas = sigmas
            self.p = p
            self.d = d

    def sample(self, n):
        """sample n samples from the mixture of gaussian

        return X: (np.array, (n, d)) data matrix from sample"""
        # Sample the component which we will draw from
        sampled_components = np.random.multinomial(1, self.p, size=(
            n,)).argmax(1)  # array indexing goes from 0 - (C -1)
        # Sample the gaussian component conditioned on sampled_components
        X = np.zeros((n, self.d))
        for i, component in enumerate(sampled_components):
            sample = np.random.multivariate_normal(mean=self.mus[component],
                                                   cov=self.sigmas[component],
                                                   size=(1,))
            X[i, :] = sample
        return X

    from sklearn.datasets import make_blobs


def make_gaussian_blobs_2d_grid(n_samples, gaussian_std,
                                xlims, ylims,
                                gridsize, equal_cluster_binning=True):
    """
    Create Gaussian mixtures with centers uniformly over a grid

    :param n_samples: (int) number of samples
    :param gaussian_std: (float) standard deviation of each gaussian
    :param xlims: ([float, float]) xmin and xmax for the grid
    :param ylims: ([flaot, float]) ymin and ymax for the grid
    :param gridsize: (int) the gridsize along x and y axis
    :param equal_cluster_binning: (bool) if true, the n_samples will be divided over the clusters equally
    """
    xmin, xmax = xlims
    ymin, ymax = ylims
    # We need dx and dy to be defined such that
    # the centers occur at regular intervals
    # from min to max with gridsize number of
    # them from start to finish (endpoints included)
    dx = float(xmax - xmin) / (gridsize - 1)
    dy = float(xmax - xmin) / (gridsize - 1)

    # Create grid of centers (note that this coordinate system has y-axis flipped)
    centers_y, centers_x = np.mgrid[slice(ymin, ymax + dy, dy),
                                    slice(xmin, xmax + dx, dx)]

    # We don't need the 2d ordering, flatten into coordinates
    # this way we can more easily iterate over them.
    centers = np.hstack((centers_x.reshape(-1, 1), centers_y.reshape(-1, 1)))

    if equal_cluster_binning:
        # Divisor and remainder. We put all leftover samples in cluster 1
        n_samples_per_cluster = n_samples // gridsize**2
        leftover_samples = n_samples % n_samples_per_cluster
        centers_repeated = np.repeat(
            centers, repeats=n_samples_per_cluster, axis=0)
        centers_to_sample = np.vstack(
            (centers_repeated, centers[0, :].reshape(1, 2).repeat(leftover_samples, axis=0)))
    else:
        # Sample n_samples uniformly from the clusters
        centers_choice_sample = np.random.multinomial(
            n_samples, pvals=np.ones(gridsize**2) / (gridsize**2))
        centers_to_sample = np.repeat(centers, centers_choice_sample, axis=0)

    assert centers_to_sample.shape == (
        n_samples, 2), "Wrong number of samples after repeating"

    X, _ = make_blobs(n_samples, centers=centers_to_sample,
                      cluster_std=gaussian_std)
    return X
