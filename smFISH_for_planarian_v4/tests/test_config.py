# run_server.py

from functions.io_utils import load_config

config = load_config("../config.yaml")  # relative path to YAML

print("smFISHChannelPath:", config["smFISHChannelPath"])
print("voxel_size:", config["voxel_size"])
print("experimentAverageThreshold:", config["experimentAverageThreshold"])
print("saveSpotInformation:", config["saveSpotInformation"])
print("plotSpotLabel:", config["plotSpotLabel"])
