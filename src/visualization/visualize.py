from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.PARAMATERS import data_experiments_dir, img_dir, project_dir


def plot_2d_mog():
    # Plot the dataset
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))

    mog_datasets_path = Path(project_dir) / 'data' / \
        'synthetic' / 'mog_datasets'

    X_spread = np.load(mog_datasets_path / 'X_2d_2mog_spread.npy')
    X_spiked = np.load(mog_datasets_path / 'X_2d_2mog_spiked.npy')

    # plot spiked
    ax[0].scatter(x=X_spiked[:, 0], y=X_spiked[:, 1])
    ax[0].set_title('spiked MoG')
    ax[1].scatter(x=X_spread[:, 0], y=X_spread[:, 1])
    ax[1].set_title('spread MoG')

    plt.tight_layout()
    fig.savefig(Path(img_dir) / 'spiked_vs_spread_mog_scatter.png')


def visualise_function(f,
                       dx=0.05, dy=0.05,
                       ylims=[-2, 2], xlims=[-2, 2]):
    """Visualise a 2d function f which takes a vector 2x1 as input

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

    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    z = z[:-1, :-1]
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    pcolor_handle = ax.pcolor(x, y, z, cmap='bwr', vmin=z_min, vmax=z_max)
    # set the limits of the plot to the limits of the data
    #ax.set_axis([x.min(), x.max(), y.min(), y.max()])
    cbar = fig.colorbar(pcolor_handle)

    return fig, ax


def visualise_sampling_grid(X_sampled, gridsize=3):
    """Show how herding algorithm samples the datapoints

    Using X_sampled, which is the original design matrix with rows
    permuted such that X_sampled[i] is the i'th row chosen by the sampling
    algorithm. This enables calculating X_sampled by arbitrary algorithms
    without having to hardcode algorithm specifics in the visualisation.

    :param X_sampled: (n x d, np.ndarray) permuted design matrix
    :param gridsize: (int) what size of the gridsize x gridsize to plot"""
    fig, ax = plt.subplots(
        gridsize, gridsize, figsize=(4 * gridsize, 4 * gridsize))

    for i in range(0, gridsize):
        for j in range(0, gridsize):
            ax[i, j].scatter(X_sampled[:, 0], X_sampled[:, 1], alpha=0.5, s=10)
            ax[i, j].scatter(X_sampled[:gridsize*i+j+1, 0],
                             X_sampled[:gridsize*i+j+1, 1], marker='^', c='red', s=100)
            ax[i, j].set_title(r'Sample n = {}'.format(gridsize*i+j+1))

    return fig, ax


def plot_learning_curves_mc_vs_kh(learning_curves_mc, learning_curve_fw,
                                  fig, ax, plot_type='plot'):
    """Plot learning curves for MC and KH

    :param learning_curves_mc: (np.ndarray, (num_curves, stop_t)) learning curves from MC
    :param learning_curve_fw: (np.ndarray, (stop_t,)) learning curve from KH
    :param fig, ax: fig and ax object
    :param plot_types: Type of plot to use, ('plot', 'semilogy', 'loglog')

    :return fig, ax:"""
    t = np.arange(1, learning_curve_fw.shape[0] + 1)
    # MC
    mc_avg = learning_curves_mc.mean(axis=0)
    mc_std = learning_curves_mc.std(axis=0)
    upper_ci = mc_avg + mc_std
    lower_ci = mc_avg - mc_std
    ax.plot(t, mc_avg, color='blue', label='MC')
    ax.fill_between(t, lower_ci, upper_ci, alpha=0.2,
                    color='blue', label='MC: 1 std CI')

    # Depending on type of plot we use different plotting arguments
    if plot_type == 'plot':
        ax.plot(t, mc_avg, color='blue', label='MC')
        ax.plot(t, learning_curve_fw, color='red', label='FW (KH)')
    elif plot_type == 'semilogy':
        ax.semilogy(t, mc_avg, color='blue', label='MC')
        ax.semilogy(t, learning_curve_fw, color='red', label='FW (KH)')
    elif plot_type == 'loglog':
        ax.loglog(t, mc_avg, color='blue', label='MC')
        ax.loglog(t, learning_curve_fw, color='red', label='FW (KH)')
    else:
        # Just do normal plot
        ax.plot(t, mc_avg, color='blue', label='MC')
        ax.plot(t, learning_curve_fw, color='red', label='FW (KH)')

    return fig, ax


def plot_learning_curves_mc_vs_kh_k_fold(learning_curves_mc, learning_curves_fw,
                                         fig, ax, plot_type='plot'):
    """Plot learning curves for MC and KH when running k_fold cv

    :param learning_curves_mc: (np.ndarray, (k_folds, num_curves, stop_t)) learning curves from MC
    :param learning_curves_fw: (np.ndarray, (k_folds, stop_t)) learning curves from KH
    :param fig, ax: fig and ax object
    :param plot_types: Type of plot to use, ('plot', 'semilogy', 'loglog')

    :return fig, ax:"""
    assert learning_curves_mc.ndim == 3
    assert learning_curves_fw.ndim == 2
    t = np.arange(1, learning_curves_fw.shape[1] + 1)
    # MC (avg)
    mc_avg = learning_curves_mc.mean(axis=1)
    mc_std = mc_avg.std(axis=0)
    mc_avg_avg = mc_avg.mean(axis=0)
    upper_ci = mc_avg_avg + mc_std
    lower_ci = mc_avg_avg - mc_std
    ax.fill_between(t, lower_ci, upper_ci, alpha=0.2,
                    color='blue', label='MC (avg): 1 std CI')

    # FW (raw)
    fw_avg = learning_curves_fw.mean(axis=0)
    fw_std = learning_curves_fw.std(axis=0)
    upper_ci = fw_avg + fw_std
    lower_ci = fw_avg - fw_std
    ax.fill_between(t, lower_ci, upper_ci, alpha=0.2,
                    color='red', label='FW (KH): 1 std CI')

    # Depending on type of plot we use different plotting arguments
    if plot_type == 'plot':
        ax.plot(t, mc_avg_avg, color='blue', label='MC')
        ax.plot(t, fw_avg, color='red', label='FW (KH)')
    elif plot_type == 'semilogy':
        ax.semilogy(t, mc_avg_avg, color='blue', label='MC')
        ax.semilogy(t, fw_avg, color='red', label='FW (KH)')
    elif plot_type == 'loglog':
        ax.loglog(t, mc_avg_avg, color='blue', label='MC')
        ax.loglog(t, fw_avg, color='red', label='FW (KH)')
    else:
        # Just do normal plot
        ax.plot(t, mc_avg_avg, color='blue', label='MC')
        ax.plot(t, fw_avg, color='red', label='FW (KH)')

    return fig, ax


def plot_learning_curves_all_algorithms_k_fold(learning_curves_mc, learning_curves_fw, learning_curves_levscore,
                                               fig, ax, plot_type='plot'):
    """Plot learning curves for MC and KH when running k_fold cv

    :param learning_curves_mc: (np.ndarray, (k_folds, num_curves, stop_t)) learning curves from MC
    :param learning_curves_fw: (np.ndarray, (k_folds, stop_t)) learning curves from KH
    :param learning_curves_levscore: (np.ndarray, (k_folds, stop_t)) learning curves from levscore
    :param fig, ax: fig and ax object
    :param plot_types: Type of plot to use, ('plot', 'semilogy', 'loglog')

    :return fig, ax:"""
    assert learning_curves_mc.ndim == 3
    assert learning_curves_fw.ndim == 2
    assert learning_curves_levscore.ndim == 2
    t = np.arange(1, learning_curves_fw.shape[1] + 1)
    # MC (avg)
    mc_avg = learning_curves_mc.mean(axis=1)
    mc_std = mc_avg.std(axis=0)
    mc_avg_avg = mc_avg.mean(axis=0)
    upper_ci = mc_avg_avg + mc_std
    lower_ci = mc_avg_avg - mc_std
    ax.fill_between(t, lower_ci, upper_ci, alpha=0.2,
                    color='blue', label='MC (avg): 1 std CI')

    # FW (raw)
    fw_avg = learning_curves_fw.mean(axis=0)
    fw_std = learning_curves_fw.std(axis=0)
    upper_ci = fw_avg + fw_std
    lower_ci = fw_avg - fw_std
    ax.fill_between(t, lower_ci, upper_ci, alpha=0.2,
                    color='red', label='FW (KH): 1 std CI')

    # levscore (raw)
    levscore_avg = learning_curves_levscore.mean(axis=0)
    levscore_std = learning_curves_levscore.std(axis=0)
    upper_ci = levscore_avg + levscore_std
    lower_ci = levscore_avg - levscore_std
    ax.fill_between(t, lower_ci, upper_ci, alpha=0.2,
                    color='green', label='Levscore (deterministic): 1 std CI')

    # Depending on type of plot we use different plotting arguments
    if plot_type == 'plot':
        ax.plot(t, mc_avg_avg, color='blue', label='MC')
        ax.plot(t, fw_avg, color='red', label='FW (KH)')
        ax.plot(t, levscore_avg, color='green',
                label='Levscore (deterministic)')
    elif plot_type == 'semilogy':
        ax.semilogy(t, mc_avg_avg, color='blue', label='MC')
        ax.semilogy(t, fw_avg, color='red', label='FW (KH)')
        ax.semilogy(t, levscore_avg, color='green',
                    label='Levscore (deterministic)')
    elif plot_type == 'loglog':
        ax.loglog(t, mc_avg_avg, color='blue', label='MC')
        ax.loglog(t, fw_avg, color='red', label='FW (KH)')
        ax.loglog(t, levscore_avg, color='green',
                  label='Levscore (deterministic)')
    else:
        # Just do normal plot
        ax.plot(t, mc_avg_avg, color='blue', label='MC')
        ax.plot(t, fw_avg, color='red', label='FW (KH)')
        ax.plot(t, levscore_avg, color='green',
                label='Levscore (deterministic)')

    return fig, ax


def plot_learning_curves_traces_all_algorithms_k_fold(learning_curves_mc, learning_curves_fw, learning_curves_levscore,
                                                      fig, ax, plot_type='plot'):
    """Plot learning curves for MC and KH when running k_fold cv (traces)

    :param learning_curves_mc: (np.ndarray, (k_folds, num_curves, stop_t)) learning curves from MC
    :param learning_curves_fw: (np.ndarray, (k_folds, stop_t)) learning curves from KH
    :param learning_curves_levscore: (np.ndarray, (k_folds, stop_t)) learning curves from levscore
    :param fig, ax: fig and ax object
    :param plot_types: Type of plot to use, ('plot', 'semilogy', 'loglog')

    :return fig, ax:"""
    assert learning_curves_mc.ndim == 3
    assert learning_curves_fw.ndim == 2
    assert learning_curves_levscore.ndim == 2

    k_folds = learning_curves_fw.shape[0]
    t = np.arange(1, learning_curves_fw.shape[1] + 1)
    # MC (avg)
    mc_avg = learning_curves_mc.mean(axis=1)

    # Depending on type of plot we use different plotting arguments
    for i in range(k_folds):
        # Depending on type of plot we use different plotting arguments
        if plot_type == 'plot':
            ax.plot(t, mc_avg[i], color='blue', label='MC')
            ax.plot(t, learning_curves_fw[i], color='red', label='FW (KH)')
            ax.plot(t, learning_curves_levscore[i], color='green',
                    label='Levscore (deterministic)')
        elif plot_type == 'semilogy':
            ax.semilogy(t, mc_avg[i], color='blue', label='MC')
            ax.semilogy(t, learning_curves_fw[i], color='red', label='FW (KH)')
            ax.semilogy(t, learning_curves_levscore[i], color='green',
                        label='Levscore (deterministic)')
        elif plot_type == 'loglog':
            ax.loglog(t, mc_avg[i], color='blue', label='MC')
            ax.loglog(t, learning_curves_fw[i], color='red', label='FW (KH)')
            ax.loglog(t, learning_curves_levscore[i], color='green',
                      label='Levscore (deterministic)')
        else:
            # Just do normal plot
            ax.plot(t, mc_avg[i], color='blue', label='MC')
            ax.plot(t, learning_curves_fw[i], color='red', label='FW (KH)')
            ax.plot(t, learning_curves_levscore[i], color='green',
                    label='Levscore (deterministic)')

    return fig, ax


def plot_learning_curves_traces_mc_vs_kh_k_fold(learning_curves_mc, learning_curves_fw,
                                                fig, ax, plot_type='plot'):
    """Plot learning curves for MC and KH when running k_fold cv

    :param learning_curves_mc: (np.ndarray, (k_folds, num_curves, stop_t)) learning curves from MC
    :param learning_curves_fw: (np.ndarray, (k_folds, stop_t)) learning curves from KH
    :param fig, ax: fig and ax object
    :param plot_types: Type of plot to use, ('plot', 'semilogy', 'loglog')

    :return fig, ax:"""
    assert learning_curves_mc.ndim == 3
    assert learning_curves_fw.ndim == 2
    k_folds = learning_curves_fw.shape[0]
    t = np.arange(1, learning_curves_fw.shape[1] + 1)
    # MC (avg)
    mc_avg = learning_curves_mc.mean(axis=1)

    # Depending on type of plot we use different plotting arguments
    for i in range(k_folds):
        if plot_type == 'plot':
            ax.plot(t, mc_avg[i], color='blue', label='MC')
            ax.plot(t, learning_curves_fw[i], color='red', label='FW (KH)')
        elif plot_type == 'semilogy':
            ax.semilogy(t, mc_avg[i], color='blue', label='MC')
            ax.semilogy(t, learning_curves_fw[i], color='red', label='FW (KH)')
        elif plot_type == 'loglog':
            ax.loglog(t, mc_avg[i], color='blue', label='MC')
            ax.loglog(t, learning_curves_fw[i], color='red', label='FW (KH)')
        else:
            # Just do normal plot
            ax.plot(t, mc_avg[i], color='blue', label='MC')
            ax.plot(t, learning_curves_fw[i], color='red', label='FW (KH)')

    return fig, ax


def visualise_mmd_curve(mmd_curve, loglog=True, ylims=[0.0, 1.0]):
    """Visualise the trace of the mmd score as a function of the size

    Given a curve which has mmd_curve[t] = MMD(P, Q_t), visualise
    both the normal plot and the log plot.

    :param mmd_curve: (np.ndarraym,(stop_t,)) array of mmd scores
    :param loglog: (bool) if true, plot loglog, else semilogy

    :return fig, ax:"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # first we do normal plot
    ax[0].plot(mmd_curve, linestyle='-')
    ax[0].set_xlabel(r't')
    ax[0].set_ylabel(r'$MMD(\hat{P}, \hat{Q}_t)$')
    ax[0].set_ylim(ylims)
    ax[0].set_title('MMD vs t')

    # NB: If we have relationships of kind MMD = O(t**-a)
    # we will have relationships of the kind
    # log(y) = log(C) - a log(t)
    # If we have relationships like MMD = A exp(-O(t))
    # we will have relationships of the kind
    # log(y) = log(A) - C * t
    # Choose the plot that makes sense given the method visualising

    # Note, we for loglog and semilogy plot, can't have y <= 0,
    # so we remove last element
    t = np.arange(0, len(mmd_curve))
    if loglog:
        ax[1].loglog(t[0:-1], mmd_curve[0:-1], linestyle='-', marker='^')
        ax[1].set_title('MMD vs t, loglog-plot')
    else:
        ax[1].semilogy(t[0:-1], mmd_curve[0:-1], linestyle='-', marker='^')
        ax[1].set_title('MMD vs t, logy-plot')

    return fig, ax
