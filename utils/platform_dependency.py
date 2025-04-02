"""
    SPecial utility needed to handle differences with 
        - Linux and Windows 
        - AMD vs Nvidia GPU 

    Code is guaranteed to work on Linux with AMD GPU and ROCm installed
    
    Lots of packages behave differently on Linux then on Windows (e.g. YOLO which does not run out of the box)
    HW support for AMD is lacking expecially on older RDNA2 GPU's (latest RDNA4 cards reportedly are better)
"""

import torch
import subprocess
import platform
import torch
import os

def get_platform(): 
    """
    Try to detect what OS you are running on.
    """
    return platform.system()


def get_gpu_info():
    """
        Try to detect the GPU platform used in the computer 
        #TODO: confirm working code on a Windows system. 
    """
    system = get_platform()
    if system == 'Linux': 
        result = subprocess.run(['lspci'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elif system == 'Windows': 
        result = subprocess.run(['wmic', 'path', 'win32_videocontroller', 'get', 'caption'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        exit('Your OS is not supported by this project.')
    gpu_info = result.stdout.decode('utf-8').lower()
    if 'nvidia' in gpu_info:
        return 'nvidia'
    elif 'amd' in gpu_info:
        return 'amd'
    else:
        return False

def yolo_override(model):
    """AMD RDNA2 and YOLO don't seem to like each other very much"
    this utility will force yolo models to be CPU inferred in stead
    of patching all kinds of issues with AMD support for YOLO: 

    arguments:
        Takes the Yolo model
    returns: 
        Yolo model set to CPU
    """
    device = torch.device('cpu')
    model.to(device)
    print('CPU override applied to model')
    return model

    
