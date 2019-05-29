"""
All algorithms used for subsampling dataset in a smart way
"""

import numpy as np


def mc_sampling(K, stop_t=None):
    """Uniformly sample from X by shuffling the indices and returning a new matrix"""
    # shuffles in place, so we need a new matrix
    if stop_t is None:
        stop_t = K.shape[0]
    sampled_order = np.random.permutation(K.shape[0])[:stop_t]
    return sampled_order


class kernel_herding:
    """
    Notation, we let Qt denote the set of sampled points after finishing iteration t,
    Ut denote the set of unsampled points after finishing iteration t. K_ab then corresponds
    to the kernel matrix by taking the inner product \Phi_a^T \Phi_b in the RKHS.
    """

    def __init__(self, K, stop_t):
        self.K = K
        self.n = K.shape[0]
        self.stop_t = stop_t
        if self.stop_t is None:
            self.stop_t = self.n
        self.sampled_order = np.zeros(stop_t)

        # These just help run the algorithm
        # These are all boolean, a one represents that
        # the x_i at that index has been sampled / unsampled respectively
        self.initial_indices = np.ones(self.n).astype(bool)
        self.sampled_indices = np.zeros(self.n).astype(bool)
        self.unsampled_indices = self.initial_indices.copy().astype(bool)
        self.arange_indices = np.arange(0, self.n).astype(bool)

        # Check for faults
        assert self.K.shape == (
            self.n, self.n), "K should be of shape ({}, {}), is of shape {}".format(self.n, self.n, self.K.shape)
        assert stop_t > 0, "stop_t needs to be positive"

    def _objective_func(self, t):
        """Calculate the objective function using sampled_indices until time t

        Note that this is a vectorised version of the one in the paper"""
        # This gets the corresponding sub-kernel matrices
        K_nUtm1 = self.K[np.ix_(self.initial_indices,
                                self.unsampled_indices)]
        K_Qtm1Utm1 = self.K[np.ix_(self.sampled_indices,
                                   self.unsampled_indices)]

        # Original factor is T + 1, but since we count from 0, need to increment by 1
        J = ((self.t + 2) / self.n) * \
            K_nUtm1.sum(axis=0) - K_Qtm1Utm1.sum(axis=0)
        assert J.shape == (self.unsampled_indices.sum(
        ),), "The output shape of J.shape should be ({},), is {} ".format(self.unsampled_indices.sum(), J.shape)
        return J

    def restart(self):
        """Start over, reinitialise everything"""
        self.t = 0
        self.sampled_order = np.zeros(self.stop_t)
        self.initial_indices = np.ones(self.n).reshape(-1, 1).astype(bool)
        self.sampled_indices = np.zeros(self.n).reshape(-1, 1).astype(bool)
        self.unsampled_indices = self.initial_indices.copy().astype(bool)
        self.arange_indices = np.arange(0, n).reshape(-1, 1).astype(bool)

    def run_kernel_herding(self):
        """Kernel herding on the empirical distribution of X through K

        Run kernel herding on the dataset (x_i)_i^n using the kernel matrix
        K represented as a numpy array of shape (n, n), where K_ij = K(x_i, x_j).
        Since the herding algorithm gives a new ordering of the dataset corresponding
        to what datapoint to include when we only return the indices of the new ordering.
        This is an array called return_order such that return_order[t] = index of x returned at
        end of t'th iteration of kernel herding.

        :param K: (np.ndarray, (n, n)) kernel matrix
        :param stop_t: (int, >0) stop running when t >= stop_t

        :return sampled_order: (np.array, (stop_t,)) the returned indices in the dataset for each t
        """

        # Initially (t=0) we sample x_0 always
        self.sampled_indices[0] = True
        self.unsampled_indices[0] = False
        self.sampled_order[0] = 0

        for t in range(1, self.stop_t):
            # The objective function of all points we can sample
            J = self._objective_func(t)
            # Get the index for the argmax
            J_argmax = J.argmax()
            # The index is not correct as we removed all indices
            # which has been sampled, so we map back to the correct index
            map_back_to_correct_index = self.arange_indices[self.unsampled_indices]
            sampled_index_at_t = map_back_to_correct_index[J_argmax]
            # Update the index arrays
            self.sampled_order[t] = sampled_index_at_t
            self.sampled_indices_t[sampled_index_at_t] = True
            self.unsampled_indices_t[sampled_index_at_t] = False

        print('Kernel herding finished')


class frank_wolfe(K, stop_t=None, rho_t=lambda t: 1/(1 + t)):
    pass


class frank_wolfe_line_search(K, stop_t=None):
    pass


class frank_wolfe_interior_point_method(K, stop_t=None):
    pass


class bayesian_quadrature(K, stop_t=None):
    pass
