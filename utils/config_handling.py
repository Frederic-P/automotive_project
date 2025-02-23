"""
    UTILITIES CLASS THAT PROVIDES CONFIG-READING CAPABILITIES TO ALL 
    SCRIPTS USED IN THE PROJECT. 
    https://docs.python.org/3/library/configparser.html
"""

import configparser
import os
from database import Database


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



def applyconf(dir): 
    # Load configuration
    config = read_config(dir)
    config.read('config.ini')
    connection_type = config['settings']['connection']
    user = config[connection_type]['user']
    pw = config[connection_type]['pw']
    host = config[connection_type]['host']
    db = config[connection_type]['db']
    port = config[connection_type].getint('port')
    db = Database(host,
                port,
                user,
                pw,
                db
                )
    db.connect()

    # Image directory
    basedir = config['settings']['image_directory']

    return [basedir, db]