import os
import yaml

# folder to load config file
config_path = 'config_files/'


# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    return config

