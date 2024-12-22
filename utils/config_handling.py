"""
    UTILITIES CLASS THAT PROVIDES CONFIG-READING CAPABILITIES TO ALL 
    SCRIPTS USED IN THE PROJECT. 
    https://docs.python.org/3/library/configparser.html
"""

import configparser
import os

def read_config(name):
    """
        Reads the configuration file and returns the configuration settings 
        to the program.

        Config files should be .ini-format compliant and need to be stored
        in the /config/ subfolder.
    """
    current_path = os.getcwd()
    config_path = os.path.join(current_path, name)
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

