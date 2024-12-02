"""
    UTILITIES CLASS THAT PROVIDES CONFIG-READING CAPABILITIES TO ALL 
    SCRIPTS USED IN THE PROJECT. 
    https://docs.python.org/3/library/configparser.html
"""

import configparser

def read_config(name):
    """
        Reads the configuration file and returns the configuration settings 
        to the program.

        Config files should be .ini-format compliant and need to be stored
        in the /config/ subfolder.
    """
    config_path = f"./config/{name}"
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

