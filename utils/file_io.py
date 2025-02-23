import shutil
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm


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