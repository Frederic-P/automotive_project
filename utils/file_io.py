import shutil
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm
import json
from pathlib import Path
from  configloader import Configloader
import pandas as pd



def path_handler(root_dir, path_as_string):
    """slave function for handling paths escaping."""
    path_as_string = path_as_string.replace('\\', '/')
    src = path_as_string.split('/')
    fq_src = os.path.join(root_dir, *src)
    return fq_src

def copy_file(src_file, dest_file):
    """
    Copy a single file from src_file to the destination file.
    """
    try:
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        # Copy the file
        shutil.copy(src_file, dest_file)
    except Exception as e:
        print(f"Error copying {src_file}: {e}")

def mt_copier(src_root, src_files, dest_root, max_workers=4):
    """
    MultiThreaded copier. 
    Copy multiple files from src_files to dest_root using multithreading.
    src_root = root directory of the source files
    src_files = list of file paths relative to src_root
    fq_trg_file = root directory of the destination files (FullyQualified Target File)
    
    **calculated value:
    dest_file = full path of the destination file (dest_root + one relative path.)
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit the copy tasks to the executor for each file in src_files
        futures = []
        for src_file in tqdm(src_files):
            src = src_file.split('\\')
            fq_src_file = os.path.join(src_root, *src)  #OK
            fq_trg_file = os.path.join(dest_root, *src)
            futures.append(executor.submit(copy_file, fq_src_file, fq_trg_file))

        # Wait for all tasks to complete
        for future in futures:
            future.result()

        
def make_absolute_path(dataframe, basedir, column_with_relative_dir, new_absolute_column, drop_relative=True): 
    """
    Utility to make relative paths in a dataframe absolute (relative paths are a project requirement (UNIX + Windows computers in this project))

    Parameters:
        dataframe (Pandas Dataframe); Dataframe to apply the mutationt ot
        basedir (string): basedirectory where your host system has the root element of the images stored
        column_with_relative_dir (string): name fo the column with the relative path (starting from the root element) of your images
        new_absolute_column (string): name for the new column to be used
        drop_relative (bool): If true will remove the relative column

    returns a dataframe with the absolute path. 
    """
    dataframe = dataframe.copy()
    dataframe[new_absolute_column] = dataframe[column_with_relative_dir].apply(lambda x: path_handler(basedir, x))
    if drop_relative and (new_absolute_column != column_with_relative_dir):
        dataframe = dataframe.drop(columns=(column_with_relative_dir))

    return dataframe

def dict_to_json_file(dictionary, dir, filename): 
    """Takes a python dictionary and dumps it to a JSON file.
    argumentes:
        - dictionary: A Python KV dictionary that can be serialized into JSON
        - dir : string: Directory where the file should be stored
        - filename: string: name of the JSON file to store the dict in.
    """
    os.makedirs(dir, exist_ok=True)
    file_path = os.path.join(dir, filename)
    with open(file_path, 'w+') as json_file:
        json.dump(dictionary, json_file, indent=4) 

def wipe_folder(folder): 
    """
        Takes a path to a folder as argument and wipes the folder. Usefull for wiping 
        old augmentation data and models of lower epochs. 

        Will then recreate the folder
        (quicker than deleting file by file or recursive operation)
    """
    config = Configloader()
    protected_folder = config.get('settings', 'image_directory')
    if folder ==  protected_folder: 
        print("can't wipe {folder}")
        return 
    if os.path.exists(folder):
        shutil.rmtree(Path(folder))
    os.makedirs(folder, exist_ok=True)

def copy_model(sourcefile, targetdir, filename): 
    """Copies a file to targetdir and renames it to a given name.
    arguments:
        - sourcefile: absolute or relative path to a file
        - targetdir: directory where to copy a file to
        - filename: new name of the copied file in targetdir
    """
    os.makedirs(targetdir, exist_ok=True)
    targetfile = os.path.join(targetdir, filename)
    shutil.copy(sourcefile, targetfile)


def write_readme(targetdir, message): 
    """Create a README.txt file in the given directory
    argument:
        - targetdir: string: path where the readme should be created
        - message: string: message to write.
    """

    os.makedirs(targetdir, exist_ok=True)
    targetfile = os.path.join(targetdir, 'README.txt')
    with open(targetfile, 'w+') as f:
        f.write(message)

def get_folder_content(dir): 
    """Lists the content of a directory according to asscending create time stamp (ctime)
    arugment: 
        dir: str: path to folder that you want to list

    returns:
        content: list of filenames with absolute path in sorted order 
                 from oldest createtime to most recent create time.
    """
    models = os.listdir(dir)
    model_paths = [os.path.join(dir, f) for f in models if os.path.isfile(os.path.join(dir, f))]
    content = sorted(model_paths, key=os.path.getctime)
    return content

def table_to_csv(table, dir, name, with_index=False): 
    """
    
    """
    table.to_csv(os.path.join(dir, name), index = with_index)