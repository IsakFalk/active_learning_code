from pathlib import Path

# can't get this shit to work
top_dir = '/home/isak/life/references/academics/phd/academic_material/project/active_learning_code/'

# Directory paths
project_dir = str(Path(top_dir).resolve())
img_dir = str(Path(top_dir).resolve() / 'reports' / 'figures')
data_external_dir = str(Path(top_dir).resolve() / 'data' / 'external')
data_synthetic_dir = str(Path(top_dir).resolve() / 'data' / 'synthetic')
data_experiments_dir = str(Path(top_dir).resolve() / 'data' / 'experiments')
