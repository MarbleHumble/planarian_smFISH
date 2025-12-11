
def detect_spots_dummy(config):
    """Dummy function to test server pipeline without running real detection"""
    print("Running dummy spot detection with the following parameters:")
    for k, v in config.items():
        print(f"{k}: {v}")