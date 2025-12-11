import yaml

def load_config(path="config.yaml"):
    """
    Load YAML config file and return a dictionary with parameters.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config