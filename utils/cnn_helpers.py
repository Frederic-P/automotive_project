import cv2
import os
import tensorflow as tf

def preprocess_image(image_path, use_bounding_box=False, bbox=None, size=64):
    # Function to preprocess image data for CNN; applies boundingvox
    """
        image_path (str)= the fully qualified path to an image on the drive
        use_bounding_box (bool) = apply bounding box or not
        bbox (list) = List with coordinates of boundingbox in shape [x1, y1, x2, y2]
        Returns an image cropped to the bounding box and resized to size * size 
    """

    img = cv2.imread(image_path)
    x1, y1, x2, y2 = bbox
    if use_bounding_box and x1 != x2:
        img = img[y1:y2, x1:x2]
    img = cv2.resize(img, (size, size))  # Resizing for CNN input
    img = img / 255.0  # Normalize pixel values (converts the RGB  to a float between 0 and 1 for each pixel)
    return img


def system_override():
    """
    My GPU (AMD Radeon RX 6700 XT) is not supported by ROCm, 
    this overrides some system variables and makes it automagically work. 
    This is a known workaround, see: 
    https://www.reddit.com/r/LocalLLaMA/comments/18ourt4/my_setup_for_using_rocm_with_rx_6700xt_gpu_on/?rdt=46886
    and 
    https://www.reddit.com/r/ROCm/comments/1dvsl1b/rocm_with_6700xt/
    """
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
    os.environ['ROCM_PATH'] = '/opt/rocm'
    print('System override applied - check if GPU is detected')

def system_pick_device():
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        device = 'GPU'
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
    else:
        device = 'CPU'

    # Print the device being used
    print(f"Using {device} for deep learning.")
    return device