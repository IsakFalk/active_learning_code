from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.PARAMATERS import img_dir, project_dir


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
