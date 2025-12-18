import yaml
import os
def load_config(path="config.yaml"):
    """
    Load YAML config file and return a dictionary with parameters.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_folder_in_same_directory(file_path: str, folder_name: str) -> str:
    """
    Creates a folder with the specified name in the same directory as the given file.
    If the folder already exists, returns the existing path.

    Args:
        file_path: Path of the reference file.
        folder_name: Name of the folder to create.

    Returns:
        The full path to the folder.
    """
    directory = os.path.dirname(os.path.abspath(file_path))
    folder_path = os.path.join(directory, folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")
    return folder_path