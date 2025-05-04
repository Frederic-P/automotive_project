# Data collecting 
This manual describes the second phase of this project: data harvesting. 

## required services: 
1) MYSQL instance with user having CRUD rights. 
2) Working python VENV with correct modules installed. 

## required configuration: 
1) MYSQL connection information needs to be provided to the application using a file called ```config/automotive.conf.ini```. To make this file, use a template file provided as ```config/sample.conf.ini```. The example file contains a brief explanation of each configuration setting. You should replace the explanation and the closing brackets *<* and *>* by the actual value you need.
2) Virtual envirment needs to be installed and working (see installation.md)

## Data harvesting order:
Data is collected from Autoscout; this online car trading platform offer access to listings from different countries through a predictable URI; there are no limitations imposed in the robots.txt file and the connection itself is fairly stable. 

The scraper is implemented as a Jupyter Notebook in the root directory of the repository (it is not an ML notebook) ```autoscout scraper.ipynb``` Which can process about 2500 listings per hour.

The scraper will download all listings for a subselection of thirty brands and store them in the folder determined by your configuration file. The substructure in that folder is: `/brandname/modelname/uuid_of_listing`. If the script terminates by system factors (failing internet connection, unscheduled reboot... ), just restart it. Inbuilt progress tracking keeps track of completed country-brand-model combinations inbetween program restarts.


## Notice: 
The data collected like this is sufficient to identify brands; to identify models, the data from Autoscout is not granular enough (generations, chassistypes) and needs to be linked to chassiscodes gather from autodoc (see: `Autodoc scraper.ipynb` in the root folder), this matching can be done rule-based but is not implemented in this project.
- You'd want to use the support features in the MYSQL-database such as (`year`, `first_registration` and `shelltype`) to match with the data gathered from Autodoc. Any kind of matching will be sensitive to errors though. 