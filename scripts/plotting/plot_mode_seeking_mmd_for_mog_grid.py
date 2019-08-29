import itertools
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.algorithms import KernelHerding, LeverageScoreSampling, MCSampling
from src.PARAMATERS import img_dir
from src.utils.utils import gaussian_kernel_matrix

np.random.seed(123)

if __name__ == '__main__':
    xx = [-1, 0, 1]
    yy = [-1, 0, 1]

    mus = list(itertools.product(xx, yy))
    mus = np.array(mus)

    samples = []
    s = 0.01

    for i in range(9):
        sample_i = np.random.multivariate_normal(
            mean=mus[i], cov=s*np.eye(2), size=50)
        samples.extend(sample_i.tolist())

        X = np.array(samples)
        K = gaussian_kernel_matrix(X, s2=0.02)

    fig, ax = plt.subplots(2, 2, figsize=(2 * 6, 2 * 6),
                           sharey=True, sharex=True)
    j = 9

    # KH
    kh = KernelHerding(K)
    kh.run()
    X_sampled_order = X[kh.sampled_order]

    ax[0, 0].scatter(X[:, 0], X[:, 1], s=20)
    ax[0, 0].scatter(X_sampled_order[:j, 0], X_sampled_order[:j,
                                                             1], s=50, zorder=10, color='red', marker='^')
    txt = [str(i+1) for i in range(j)]
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    for i in range(j):
        ax[0, 0].annotate(txt[i], (X_sampled_order[i, 0], X_sampled_order[i, 1]),
                          color='black', fontsize=24, bbox=bbox_props)
    ax[0, 0].set_title('FW-kh', fontsize=30)

    # MC
    mc = MCSampling(K)
    mc.run()
    X_sampled_order = X[mc.sampled_order]
    ax[0, 1].scatter(X[:, 0], X[:, 1], s=20)
    ax[0, 1].scatter(X_sampled_order[:j, 0], X_sampled_order[:j,
                                                             1], s=50, zorder=10, color='red', marker='^')
    txt = [str(i+1) for i in range(j)]
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    for i in range(j):
        ax[0, 1].annotate(txt[i], (X_sampled_order[i, 0], X_sampled_order[i, 1]),
                          color='black', fontsize=24, bbox=bbox_props)
    ax[0, 1].set_title('MC', fontsize=30)

    # Levscore
    lev_det = LeverageScoreSampling(
        K, tau=0.02, stop_t=None, greedy=True)
    lev_det.run()
    X_sampled_order = X[lev_det.sampled_order]
    ax[1, 0].scatter(X[:, 0], X[:, 1], s=20)
    j = 9
    ax[1, 0].scatter(X_sampled_order[:j, 0], X_sampled_order[:j,
                                                             1], s=50, zorder=10, color='red', marker='^')
    txt = [str(i+1) for i in range(j)]
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    for i in range(j):
        ax[1, 0].annotate(txt[i], (X_sampled_order[i, 0], X_sampled_order[i, 1]),
                          color='black', fontsize=24, bbox=bbox_props)
    ax[1, 0].set_title('LS-det', fontsize=30)

    # Levscore
    lev_rand = LeverageScoreSampling(
        K, tau=0.02, stop_t=None, greedy=False)
    lev_rand.run()
    X_sampled_order = X[lev_rand.sampled_order]
    ax[1, 1].scatter(X[:, 0], X[:, 1], s=20)
    j = 9
    ax[1, 1].scatter(X_sampled_order[:j, 0], X_sampled_order[:j,
                                                             1], s=50, zorder=10, color='red', marker='^')
    txt = [str(i+1) for i in range(j)]
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    for i in range(j):
        ax[1, 1].annotate(txt[i], (X_sampled_order[i, 0], X_sampled_order[i, 1]),
                          color='black', fontsize=24, bbox=bbox_props)
    ax[1, 1].set_title('LS-rand', fontsize=30)

    fig.savefig(Path(img_dir) / "kh_is_mode_seeking_mog",
                dpi=600, bbox_inches='tight')
    plt.close(fig)
