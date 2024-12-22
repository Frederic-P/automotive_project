"""
utilities to handle file input/output when using batches.
"""

import shutil
from concurrent.futures import ThreadPoolExecutor
import os 
from tqdm import tqdm 

def copy_file(src_file, dest_dir):
    """
    Copy a single file from src_file to the destination directory.
    """
    try:
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
        # Copy the file
        shutil.copy(src_file, dest_dir)
    except Exception as e:
        print(f"Error copying {src_file}: {e}")

def mt_copier(src_root, src_files, dest_root, max_workers=4):
    """
    MultiThreade_copier. 
    Copy multiple files from src_files to dest_dir using multithreading.
    src_root = root directory of the source files
    src_files = list of file paths relative to src_root
    dest_root = root directory of the destination files
    
    **calculated value:
    dest_file = full full path of the destination file (dest_root + one relative path.)
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit the copy tasks to the executor for each file in src_files
        futures = []
        for src_file in tqdm(src_files):
            fq_src_file = os.path.join(src_root, src_file)
            fq_trg_file = os.path.join(dest_root, src_file)
            futures.append(executor.submit(copy_file, fq_src_file, fq_trg_file))

        # Wait for all tasks to complete
        for future in futures:
            future.result()