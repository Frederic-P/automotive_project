import cv2
import os
import tensorflow as tf
import numpy as np
import random 
import shutil


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


def augm_channel_shuffle(image):
    """
    AUGMENTATION FUNCTION: 

        Takes an image and shuffles the RGB channels

        Parameter:
            Image (as numpy array)

        Returns:
            an  image with shuffled color channels.
    """
    if image.shape[2] == 3:  #requires RGB channel
        channels = np.split(image, 3, axis=2)  # Split the image into 3 channels (R, G, B)
        np.random.shuffle(channels)  # Shuffle the channels
        shuffled_image = np.concatenate(channels, axis=2)  # Recombine the shuffled channels back into one image
        return shuffled_image
    else:
        raise ValueError("Input image must have 3 channels (RGB)")
    

def augm_add_random_noise(image, mean=0, sigma=5):
    """
    AUGMENTATION FUNCTION: 
        Adds Gaussian noise to an image.

        Parameter:
            Image (numpy array): Input image.
            Mean (int): Mean of the Gaussian noise.
            Sigma (float): Standard deviation of the Gaussian noise.

        Returns:
            Image with noise
    """
    image_float = image.astype(np.float32)
    gaussian_noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = image_float + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image

def augm_stretch_image_and_bbox(image, bbox, max_stretch=0.10): 
    """
    AUGMENTATION FUNCTION: 
        Stretches the image up to a given percentage (max_stretch) and updates the bounding box coordinates.
        
        Parameter:
            image (numpy array): Input image.
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2) for the top-left and bottom-right corners (IMPORTANT to use this order in this tuple!!!)
            max_stretch (float): The maximum stretch as a percentage

        Returns:
            stretched_image (numpy array): Stretched image.
            updated_bbox (tuple): Updated bounding box coordinates (x1, y1, x2, y2).
        IF your original BBOX were four NaNs, then this function will return four nans
    """

    h, w = image.shape[:2]
    
    # Calculate stretch factor for both X and Y axes (up to 10%)
    stretch_x = random.uniform(1, 1 + max_stretch)  # Random factor between 1 and (1 + max_stretch)
    stretch_y = random.uniform(1, 1 + max_stretch)  # Same for Y axis

    # Resize the image based on the stretch factors
    new_w = int(w * stretch_x)
    new_h = int(h * stretch_y)
    stretched_image = cv2.resize(image, (new_w, new_h))

    # Update the bounding box coordinates
    x1, y1, x2, y2 = bbox
    if all(np.isnan(b) for b in bbox):
        updated_bbox = (np.nan, np.nan, np.nan, np.nan)
    else:
        new_x1 = int(x1 * stretch_x)
        new_y1 = int(y1 * stretch_y)
        new_x2 = int(x2 * stretch_x)
        new_y2 = int(y2 * stretch_y)

        updated_bbox = (new_x1, new_y1, new_x2, new_y2)

    return stretched_image, updated_bbox


# def load_data(df, use_bounding_boxes=False):
#     images = []
#     for _, row in tqdm(df.iterrows()):
#         bbox = (row['yolobox_top_left_x'], row['yolobox_top_left_y'], row['yolobox_bottom_right_x'], row['yolobox_bottom_right_y'])
#         img = preprocess_image(row['abs_image_path'], use_bounding_boxes, bbox)
#         images.append(img)
#     return np.array(images)


def cast_bbox_values(df, bboxcolumns= ['yolobox_top_left_x', 'yolobox_top_left_y', 'yolobox_bottom_right_x', 'yolobox_bottom_right_y']):
    df = df.copy()
    df = df.fillna('-1')  #to suppres a futurewarning in pandas.
    df[bboxcolumns] = df[bboxcolumns].astype(int)
    return df


def image_generator(batch_size, data_frame, bboxs, shape, y_train_encoded):
    """
        A generator function that provide JIT image preprocessing 
        for your neural networks. This should allow for some kind of 
        Out Of Core learning.

        This generator will not work when aliassing on sql column names is used; doesn't matter for now. 

        Parameters: 
            batch_size (INT): HWat's the batch size your CNN works with (so far I always used 32)
            data_frame (Pandas df): The Pandas DataFrame that holds the data you want to train on. NOTE: do NOT alias the bbox columns!!
            bboxs (Bool): To crop or not to crop (that's the question)
            shape (INT): Size on a single dimension of your intended numpy raster
            y_train_encoded (pandas series); the series of encoded labels should be of equal shape as data_frame

        Returns: 
            preprocessed images as a numpy array according to given configuration (shape/bbox)
            target labels as a numpy array (encoded)
    """
    assert(len(y_train_encoded) == len(data_frame))
    while True:  # Loop forever so the generator never terminates
        batch_images = []
        batch_labels = []
        for i in range(batch_size):
            idx = np.random.randint(0, len(data_frame))
            row = data_frame.iloc[idx]
            coords = [row['yolobox_top_left_x'], row['yolobox_top_left_y'], row['yolobox_bottom_right_x'], row['yolobox_bottom_right_y']]
            img = preprocess_image(row['abs_path'], bboxs, coords, shape)
            batch_images.append(img)
            batch_labels.append(y_train_encoded[idx])  # Make sure y_train_encoded corresponds to the correct index
        yield np.array(batch_images), np.array(batch_labels)



def shuffle_df(df, rs = False): 
    """takes a dataframe (df) and random state value (int) and returns a shuffled copy of your given dataframe."""
    df = df.copy()
    if rs == False:
        return df.sample(frac=1).reset_index(drop=True)
    else: 
        return df.sample(frac=1, random_state=rs).reset_index(drop=True)


def ordinal_encoder_to_dict(oe): 
    """takes an ordinal encoder and returns a python Dictionary with integer keys and label values"""
    labels = oe.categories[0]    
    integers = oe.transform([[x] for x in labels]).flatten()
    d = {}
    for k,v in zip(integers, labels):
        d[int(k)] = v
    return d


def load_one_model(dir, extension = '.keras'):
    """
    Loads a model with a specific extension for a directory. This method is only 
    intended for directories with a single model stored in them.
    """
    models = [f for f in os.listdir(dir) if f.endswith(extension)]
    assert len(models) > 0
    model = models[0]
    modeldir = os.path.join(dir, model)
    loaded_model = tf.keras.models.load_model(modeldir)
    return loaded_model