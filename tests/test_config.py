from functions.io_utils import load_config
from pathlib import Path
import os

# Optional: ensure current working directory is project root
os.chdir(Path(__file__).resolve().parent.parent)

# Load test config
config = load_config("config_example.yaml")

# Print all parameters for verification
print("Loaded config parameters:")
for k, v in config.items():
    print(f"{k}: {v}")

# Optional: run a small test function (dummy spot detection)
from functions.spot_detection import detect_spots_dummy  # create a dummy function for testing
from functions.io_utils import create_folder_in_same_directory

results_folder = create_folder_in_same_directory(config["smFISHChannelPath"], config["resultsFolderName"])
