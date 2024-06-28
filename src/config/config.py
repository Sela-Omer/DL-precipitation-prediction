import configparser
import os.path

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the config.ini file
config.read('src/config/config.ini')

if 'OVERWRITE_CONFIG_PATH' in config['APP'] and os.path.isfile(config['APP']['OVERWRITE_CONFIG_PATH']):
    print(f'overwriting config with {config["APP"]["OVERWRITE_CONFIG_PATH"]}')
    config.read(config['APP']['OVERWRITE_CONFIG_PATH'])
