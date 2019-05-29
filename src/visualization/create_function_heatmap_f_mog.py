import numpy as np

from src.PARAMATERS import img_dir, project_dir, s2
from src.utils.utils import create_f
from src.visualization import visualise_function

f_param_dir = project_dir / 'data' / 'synthetic' / 'mog_datasets' / 'mog_f'

if __name__ == '__main__':
    # Load saved function parameters
    x_is = np.load(f_param_dir / 'x_is.npy')
    alpha_is = np.load(f_param_dir / 'alpha_is.npy')

    # Create function and plot
    f = create_f(x_is, alpha_is, s2)
    fig, ax = visualise_function(f)
    fig.savefig(img_dir / 'f_mog_heatmap.png')
