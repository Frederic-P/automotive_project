"""
    This script is intended to be triggered by CRON jobs (Linux) to fetch the most recent listings 
    from autoscout. It implements a trie class to see if the listings are new or not.
    The reasoning behind trie-classes is explained in /experimental_code/trie.ipynb

    It should only be run once the initial database is populated. (i.e. the autoscout scraper.ipynb ran completely)

    countries and brand list is extracted this time from the MYSQL database.
"""
# IMPORT SECTION
from bs4 import BeautifulSoup
import json
import pandas as pd
import requests
import numpy as np
from tqdm import tqdm
import sys
import time
sys.path.append('utils')
import os
import config_handling as conf
from multithread_image_ripper import download_images
from database import Database

#DATABASE SECTION
# Connect to database
config = conf.read_config('config/automotive.conf.ini')
config.read('config.ini')
connection_type = config['settings']['connection']
connection_type
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
db.start_transaction()

#CONFIG SECTION 
#image directory: 
basedir = config['settings']['image_directory']

#QUERY WRITING SECTION
country_query = "SELECT DISTINCT countrycode FROM listings;"
brand_query = "SELECT DISTINCT brand FROM listings;"

#QUERY EXECUTION SECTION
countries = db.execute_query(country_query)
brands = db.execute_query(brand_query)
print(countries)

