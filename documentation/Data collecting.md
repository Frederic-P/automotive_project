# Data collecting 
This manual describes the second phase of this project: data harvesting. 

## required services: 
1) MYSQL instance with user having CRUD rights. 

## required configuration: 
1) MYSQL connection information needs to be provided to the application using a file called ```config/automotive.conf.ini```. To make this file, use a template file provided as ```config/sample.conf.ini```. The example file contains a brief explanation of each configuration setting. You should replace the explanation and the closing brackets *<* and *>* by the actual value you need.
2) Virtual PIP envirment needs to be installed and working (see installation.md)

## Data harvesting order:
There are two distinct phases in the data harvesting part. First of all a base collection has to be harvested. A base collection is all listings of a given list of brands available for sale in a given country. Once a basecolection for one country is 'harvested', it can be used to train the car angle detector (separate repository see: TODO)

For this first phase, set up and run the ```autoscout scraper.ipynb``` notebook. Depending on the bandwidth you should be able to process about 2500 listings per hour.