"""
    UTILITIES CLASS THAT PROVIDES CONFIG-READING CAPABILITIES TO ALL 
    SCRIPTS USED IN THE PROJECT. 
    https://docs.python.org/3/library/configparser.html

    The configloader class is configurable by setting the constant 
    PATH_TO_CONF to the path of the config file from the class itself
    (configloader.py). 
    THe class provices a simple .get() method taking section and key 
    to retrieve a config value.

    We choose to set the path to config file relative to the configloader.py
    file itself, so that the config file can be moved to a different location
    and will only need to be updated in one place.

"""

import os
import configparser

class Configloader: 
    def __init__(self):
        PATH_TO_CONF = '../config/automotive.conf.ini'

        self.path = os.path.abspath(__file__)
        self.conffile = os.path.join(os.path.dirname(self.path), PATH_TO_CONF)
        #reading configuration: 
        config = configparser.ConfigParser()
        config.read(self.conffile)
        self.config = config

    def get_section(self, section): 
        """
        Returns the section of the config file with all child kv's
        ARGUMENTS:
            section (str) = Name of the section used in the conf.ini file.
        RETURNS: 
            returns a SectionProxy object.
        """
        return self.config[section]

    def get(self, section, name): 
        """
        Returns the value for config[section][name] to the calling instance

        ARGUMENTS:
            section (str) = Name fo the section used in the conf.ini file.
            name (str) = key name of the key-value pair in the section.
        RETURNS: 
            returns the string value for the requested section/name pair in config file
        """
        return self.get_section(section)[name]
