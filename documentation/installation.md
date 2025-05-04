# Installing virtual environment. 
This guide is focussed on Ubuntu environments with AMD GPU's. Your mileage may vary using NVIDIA or other OS systems. It's not recommended to run the combination of an AMD GPU on Windows. This is a bare minimum installation guide, if you're using port forwarding or multiple computers to access the database, then you'll need to to configure certain elements differently. 

## Required components:
1) Python 3.12
2) PIP
3) MYSQL8
4) ROCM (for AMD systems)

## Installing MYSQL: 
MYSQL8 needs to be installed and configured on the system; you can follow [https://www.geeksforgeeks.org/how-to-install-mysql-on-linux/](this guide). 

Once installed at a designated user that has CRUD rights on a designated database where you'll project related tabular data. If you plan on exposing the MYSQL databse to external traffic, don't forget to open the remote access as far as needed, yet as tight as possible. Drop and truncate rights are not required. 

To create all necessary tables for the project - import the ddl.sql file; this will make sure all necessary tables and columns are there to process the data. 

## Installing Python 3.12
Ubuntu comes with a standard version of Python, it's recommended to update to Python 3.12. Be aware of this or consider changing the alias of the python3 command to the 3.12 version. You don't need to update the Alias if you work with a virtual environment. 

Pip is installed alongside Python 3.12

## Driver warnings
Make sure the correct GPU drivers are installed on the system. For AMD the default drivers that are included in Ubuntu 24.0.4 LTS are fine. 

## Requirements.txt
The code was written for a system using Linux + ROCM6.3 + Radeon RX6700XT this has some consequences for the requirements.txt file. As PIP does not provide the ROCM packages listed in the requirements.txt file, these have to be downloaded and installed manually according to the official guide by AMD: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/ The bare-metal installation is recommended over Docker. 
1) ON Linux-AMDGPU systems follow the rocm installation guide at amd.com: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/tensorflow-install.html

2) On non-linux / non-amd systems: Good luck - I can imagine you will need to replace `rocm` suffixes and specific packages but I had no platform to test this on. 

In case pip autoloads a module that is not compatible with ROCM (such as: https://pypi.org/project/tf-keras/ which has built in support for transformers and looked cool to use but broke keras), you can restore the venv by using `pip install tensorflow-rocm==2.17 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3 --upgrade  --force-reinstall ` (apply the correct Major, Minor version code for rocm in the URI parameters.)

On Windows systems you can use the requirements_windows.txt file and install an environment from there. This is the environment dumped from a system that did not have GPU-acceleration.


## Project configuration
Local variables, MYSQL user accounts, passwords etc... are handled by a config file, this config file is excluded from the VSC-tool and as such you'll need to manually configure your system. You can use the provided `sample.conf.ini` file as a guide to set it up; do not change the keynames in the sample file only provide values. The config file with your actual configuration values in should be named `automotive.conf.ini` and should be stored in the `/config` dir starting from the root of this repo. 

Some guidelines to keep in mind when setting up this config file: 
- strings are unquoted and case sensitive
- The .ini file is subdivided in sections (marked with []), do not move values from one section to another section
- Pay close attention to the use of absolute or relative paths. If a path is documented a relative, it should be written down as a subfolder to an absolute path. The documentation below identifies a kv pair as section.key = explain the expected value.

### Configuration settings explained: 

```
    [remote]
    host = <ip_of_mysql_server>
    port = <int: port of mysql server>
    user = <str: username>
    pw = <str: password>
    db = <str: database name>

    [local]
    host = localhost
    port = 3306
    user = root
    pw = VLmP78dQ1
    db = automotive

    [settings]
    connection = <one of the two above keys: remove or local>
    image_directory = <string starting from root where your images are stored. This folder will contain the augmented data when created and one folder per downloaded brand.>
    vit_configurations = <Relative path to the vit_configs.json file holding the VIT-profiles to test in notebook 3.5, if you don't make changes you can use `config/vit_configs.json`>


    [directories]
    root_dir = C:\python Projects\automotive_project
    yolo_path = models/yolo/yolov8n.pt
    final_models_dirname = final models
    binary_dir = models/bin_models
    vit_dir = models/vision_transformer
    final_angle_dir = angles
    brandphase = brand
    angled = angle_based

    [dir_augmentations]
    subfolder = augmentated data
    csv_dir = CSV-data

    [flask_angletagger]
    country_only = <Countrycode of a country that's fully donwloaded >
    username = <Username for password protected flask app>
    password = <Password for flask app login. >

    [csv_names]
    brand_validation = <name of file that tracks validationdata WITH augmentations e.g. validationdata_brandphase.csv>
    brand_test = <name of file that tracks testdata WITH augmentations e.g. testdata_brandphase.csv>
    brand_train = <name of file that tracks traindata WITH augmentations e.g. traindata_brandphase.csv>
    bin_name = <name of file for binary modeltracking: e.g. chosen_models.csv>

```


- remote.host = IP address that allows you to reach your MYSQL server; your server will need to accept remote connections and have a user account associated to it that can log in remotely (%)
- remote.port = integer that tells what port to connect to (typically 3306)
- remote.user = string username for MYSQL (needs CRU rights) remotely: you are NOT encouraged to use the root account here!!!
- remote.pw = string: Case sensitive password for the user account
- remote.db = string: Name of the database (if you use the provided .sql dump) it is `automotive`
- local.host = string or ip used to identify the localhost (127.0.0.1 or localhost) are typically used here
- local.port = integer that tells what port to connect to (typically 3306)
- local.user = string username for MYSQL (you can use the root account here)
- local.pw = string: Case sensitive password for the user account
- local.db = string: Name of the database (if you use the provided .sql dump) it is `automotive` (is shared with remote!!) 
- settings.connection = string = Name of the connection configuration to load: either `remote` or `local`
- settings.image_directory = string starting from the system root (LINUX: starting with `/`; Windows starting with Letter `Z:\`)
- settings.vit_configurations = relative path starting from the value set for `directories.root_dir` where the VIT-config file is saved. If you are using the config file from the Git repo, you can use `config/vit_configs.json` as a value.
- directories.root_dir = string = absolute path to where you have cloned this repository.
- directories.yolo_path = string = relative path; starting from directories.root_dir where you will download and read the Yolo model from
- directories.final_models_dirname = relative path starting from directories.root_dir where final models will be stored and read from. 
- directories.binary_dir = relative path starting from directories.root_dir where binary models are trained for (usability tagging of images)
- directories.vit_dir = relative path starting from directories.root_dir where VIT-models and checkpoints are kept. 
- directories.final_angle_dir = relative path starting from directories.root_dir where angle models are kept. 
- directories.brandphase = leave this at 'brand'
- directories.angled foldername to stare angle prediciton models: leva at 'angle_based'
- dir_augmentations.subfolder =String = name of the augmentations subfolder.
- dir_augmentations.csv_dir = String = name of the subvolder where augmentation csv files are stored. 
- flask_angletagger.country_only = String = Countrycodeletter: B,D,F,NL are options you can use. The angle tagger will then only show listings from this country. See the FlaskTagging.md file as to why this matters.
- flask_angletagger.username = string = username to access the Flask taggin application.
- flask_angletagger.password = string = password to access the Flask tagging application (put in a strong password if you use portforwarding i.e. Telebit), otherwise this does not really matter.
- csv_names.brand_validation = Name of the csv-file generated by to_colab.ipynb for the validationdata.
- csv_names.brand_test = Name of the csv-file generated by to_colab.ipynb for the testdata.
- csv_names.brand_train = Name of the csv-file generated by to_colab.ipynb for the trainingdata.
- csv_names.bin_name = Name used of the csv-file to track the four selected binary models.