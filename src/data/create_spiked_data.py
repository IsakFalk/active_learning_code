from pathlib import Path

import numpy as np

from src.PARAMATERS import project_dir
from src.utils.distributions import MoG

if __name__ == '__main__':

    # Spiked dataset
    mus = np.array([[-1, -1], [1, 1]])
    sigmas = np.zeros((2, 2, 2))
    sigmas[0] = 0.2 * np.eye(2)
    sigmas[1] = 0.2 * np.eye(2)
    p = np.array([0.5, 0.5])
    mog_spiked = MoG(d=2, mus=mus, sigmas=sigmas, p=p)

    # Sample n points
    n = 500  # Should refactor this to command line configurable
    X_spiked = mog_spiked.sample(n)

    # Save datasets
    mog_data_path = Path(project_dir) / 'data' / 'synthetic' / 'mog_datasets'
    if not mog_data_path.is_dir():
        mog_data_path.mkdir()

    np.save(str(mog_data_path / 'X_2d_2mog_spiked.npy'), X_spiked)
