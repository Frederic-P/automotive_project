# installing virtual environment. 
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
