import pickle
from pathlib import Path

import numpy as np

from src.PARAMATERS import k, project_dir, s2

if __name__ == '__main__':
    # We can't pickle the function so we just recreate it when we need it
    x_is = 2 * np.random.rand(k, 2) - 1  # uniform over [-1, 1]
    alpha_is = 4 * np.random.randn(k, 1)

    # Save function paramteres
    mog_f_path = Path(project_dir) / 'data' / \
        'synthetic' / 'mog_datasets' / 'mog_f'
    if not mog_f_path.is_dir():
        mog_f_path.mkdir()

    np.save(str(mog_f_path / 'x_is.npy'), x_is)
    np.save(str(mog_f_path / 'alpha_is.npy'), x_is)
