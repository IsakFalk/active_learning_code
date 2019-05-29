import numpy as np


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
