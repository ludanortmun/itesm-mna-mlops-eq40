import yaml


def load_params(path):
    with open(path, 'r') as f:
        params = yaml.safe_load(f)
    return params