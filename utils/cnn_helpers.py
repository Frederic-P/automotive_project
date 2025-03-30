import cv2
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
import numpy as np
import pandas as pd
import random 
import shutil
import json
from sklearn.model_selection import train_test_split
import uuid
from datetime import datetime


class OptimizerStateSaver(Callback):
    #https://keras.io/guides/writing_your_own_callbacks/ 
    def __init__(self, storagefolder, phase=0):
        super().__init__()
        self.storagefolder = storagefolder
        self.phase = phase #0 = Frozen base; 1 = unfrozen base

    def on_epoch_end(self, epoch, logs=None):
        # Increment the iteration counter at the end of each epoch
        print(f"Saving optimizer state for phase {self.phase} - epcoh {epoch}.")
        save_optimizer_state(self.model, self.storagefolder, f'checkpoint_epoch_{epoch}', self.phase)



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


def augm_mirror_image(source, targetpath, name, given): 
    """
    AUGMENTATION FUNCTION: 
        helps to provide more class balance by mirroring an image over the y-axis. 
        Is intended to be used to help boost class balances for predicting brands
        and models. It does not follow the same desing preinciple as the other
        augmentation functions because of that. 
        DOES NOT STORE THE IMAGE

    Parameter: 
        source: str: Fully qualified path to the image.
        targetpath: str: Fully qualified path to the folder where augmentated data is store.
        name: str: Name of the augmentated file. 
        given: str: label of the current image

    returns:
        new_label: str: new label (right/left.)
    """
    if given == 'left':
        new_label = 'right'
    elif given == 'right':
        new_label = 'left'
    else:
        raise ValueError(f"Mirroring of images with label {given} is not allowed.")
    img = cv2.imread(source)
    mirrored_img = cv2.flip(img, 1)
    os.makedirs(targetpath, exist_ok=True)
    cv2.imwrite(os.path.join(targetpath, name), mirrored_img)
    return new_label

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


def train_test_val_splitter(df, trainratio, testratio, valratio, stratcols = False): 
    """takes a pandas df and splits it in three dfs according to the 
    given ratios. Optionally it uses tratified splitting columns defined in stratcols.
    arguments: 
    df (pandas dataframe)
    trainratio; (int): how many percent of df should end up in trainset
    testratio: (int): how many percent of df should end up in testset
    valratio: (int): how many percent of df should end up in valratio
    stratcols: (list): list of column names to perform a stratified split on. 
    """
    assert(trainratio+testratio+valratio == 100)
    if stratcols:
        train_data, remaining_data = train_test_split(
            df, test_size=(100 - trainratio) / 100, stratify=df[stratcols], random_state=42
        )
        test_data, val_data = train_test_split(
            remaining_data, test_size=valratio / (testratio + valratio), stratify=remaining_data[stratcols], random_state=42
        )
    else:
        train_data, remaining_data = train_test_split(
            df, test_size=(100 - trainratio) / 100, random_state=42
        )
        test_data, val_data = train_test_split(
            remaining_data, test_size=valratio / (testratio + valratio), random_state=42
        )
    return train_data, test_data, val_data


def augment(sampled_df, augment_dir): 
    """
        Takes a pandas dataframe containing a single row and performs a random (set of)
        augmentations on it. 

        Parameters:
            sampled_df (df): a pandas dataframe of whicht the first row only will be processed!!
            augment_dir str: Fully qualified path of the directory where the augmented file will be stored. 
    """
    row = sampled_df.iloc[0].copy()
    impath = row['abs_path']
    uuid_v4 = str(uuid.uuid4())
    new_name = uuid_v4 + '.' + impath.split('.')[-1]
    coords = (row['yolobox_top_left_x'], row['yolobox_top_left_y'], row['yolobox_bottom_right_x'], row['yolobox_bottom_right_y'])
    image = cv2.imread(impath)
    augmented_path = os.path.join(augment_dir, new_name)
    os.makedirs(os.path.dirname(augmented_path), exist_ok=True)

    base_action = random.randint(0,100)
    #80% of images has only ONE mutation
    #15% has TWO mutations
    # 5% of images has THREE mutatations applied. 
    mutation_options = [0, 1, 2]
    if base_action < 85: 
        mutations = random.sample(mutation_options, 1)         #pick ONE mutation
    elif base_action < 95: 
        mutations = random.sample(mutation_options, 2)         #pick TWO mutations
    else:
        mutations = random.sample(mutation_options, len(mutation_options))              #Perform ALL THREE mutations in random order.
    modded_img = image.copy()
    for mutation in mutations: 
        if mutation == 0: 
            modded_img = augm_channel_shuffle(modded_img)
        elif mutation == 1:
            sigma = random.randint(40, 80)
            modded_img = augm_add_random_noise(modded_img, 0, sigma)
        else:
            stretch = random.uniform(0.8, 1.3)
            modded_img, coords = augm_stretch_image_and_bbox(modded_img, coords, stretch)
    #print(type(modded_img))
    #print(modded_img.shape())
    cv2.imwrite(augmented_path, modded_img)
    row['abs_path'] = augmented_path
    row['yolobox_top_left_x'] = coords[0]
    row['yolobox_top_left_y'] = coords[1]
    row['yolobox_bottom_right_x'] = coords[2]
    row['yolobox_bottom_right_y'] = coords[3]
    return row  

def datetimestring():
    current_datetime = datetime.now()
    return current_datetime.strftime('%d-%m-%Y %H:%M:%S')

def write_msg_to_log(message, logpath):
    """
        Writes a message (str) to a file (logpath) with the datetimestring
        quickly implemented to identify training bottleneck in between epochs. 
        I'm currently suspecting the checkpoints, which can be commented out
        or a slowdown in the validation phase - which is something I'll have 
        to learn to live with.
        arguments:
            message = str: 'The message to write to a logfile.
            logpath = str: absolute path to a logfile. path must exist, 
            doesn't matter if the file exists.
    """
    with open(logpath, 'a+', encoding='utf8') as f:
        message = f'{datetimestring()} - {message}'
        f.write(message)
        f.write('\r\n')


def reducer(df, max_samples, by):
    """Training gets stuck when using the testdata during the traininprocess
    as the models get more complex; in stead of using the full testset, this 
    function will subsample it in smaller sections using a random selection
    It'll keep procentually more of smaller 'by' values 
    
    """
    sampled = []
    for _, group in df.groupby(by):
        if max_samples > len(group):
            samplesize = len(group)
        else:
            samplesize = max_samples
        small_df = group.sample(samplesize)
        sampled.append(small_df)
    small_df =  pd.concat(sampled).reset_index(drop=True)
    small_df = shuffle_df(small_df)
    return small_df


def save_optimizer_state(model, storagefolder, epoch, frozenstate):
    """Save optimizer state (learning rate) to a file.
    

        //BUG: modelcheckpoint generates name of dump. need it here to link the json dump
        with the right model.  //PATCHED - pending test
    """
    print(f'received epcoh {epoch} and state {frozenstate}')

    optimizer_config = model.optimizer.get_config()
    name = f'lr_optimizer__state_{frozenstate}__epoch_{epoch}.json'
    with open(os.path.join(storagefolder, name), 'w') as f:
        json.dump(optimizer_config, f)

def load_optimizer_state(model, storagefolder, epoch, frozenstate):
    """Restore optimizer state (learning rate) from a file.
        WARNING: ONLY USE WITH ADAMW!!

    //BUG: related to issue in load_optimizer_state: you should load the 
    json file with the highest value in the name.
    //PATCH implemented, test needed.

    #TODO!
    """
    name = f'lr_optimizer__state_{frozenstate}__epoch_{epoch}.json'
    optimizer_state_file = os.path.join(storagefolder, name)
    if os.path.exists(optimizer_state_file):
        with open(optimizer_state_file, 'r') as f:
            optimizer_config = json.load(f)
            print("Configuration or lr scheduler:")
            print(json.dumps(optimizer_config, indent=4)) 
        optimizer = AdamW.from_config(optimizer_config)  # Reinitialize optimizer with saved state
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Optimizer state restored.")
    else:
        print("No optimizer state file found. Using default optimizer settings.")



def resnet_learner(X_train, X_test, y_train, y_test, shape, apply_crop, storagefolder, batchsize, max_epochs, learning_rate = 0.001):
    #0.001 learning rate is standard for adam(w) #SRC: https://keras.io/api/optimizers/adamw/
    train_gen = image_generator(
        batch_size=batchsize, 
        data_frame=X_train, 
        bboxs=apply_crop, 
        shape=shape, 
        y_train_encoded=y_train
    )
    #Split validation separately
    val_gen = image_generator(
        batch_size=batchsize, 
        data_frame=X_test, 
        bboxs=apply_crop, 
        shape=shape, 
        y_train_encoded=y_test
    )


    #check if we have a checkpoint: 
    checkpoint_files = [f for f in os.listdir(storagefolder) if 'dump' in f and f.endswith('.keras')]
    checkpoint = False
    if len(checkpoint_files) > 0:
        #google drive will return the last created checkpooint if you use -1
        checkpoint = checkpoint_files[-1]
        print(f'CHECKPOINT found')

    if not checkpoint:
        print('NO checkpoint file present, starting from a base ResNet model')
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(shape, shape, 3))
        base_model.trainable = False
        model = models.Sequential([
            base_model,  # resnet50
            layers.GlobalAveragePooling2D(), 
            layers.Dense(256, activation='relu'),
            layers.Dense(y_train.nunique(), activation='softmax') 
        ])
        #sparse_categorical_crossentropy no need to use OHE with sparse_categorical_crossentropy!!!!
        model.compile(optimizer=AdamW(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else: 
        model = tf.keras.models.load_model(os.path.join(storagefolder, checkpoint))
        base_model = model.layers[0]    # now you know if the layer was frozen before or not.

    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=6, 
        restore_best_weights=True
    )
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-6, 
        verbose=1
    )
    model_checkpoint = ModelCheckpoint(
        os.path.join(storagefolder, 'disposable_dump_of_brand_model_epoch_{epoch:02d}__val_loss_{val_loss:.4f}.keras'),
        #'model_epoch_{epoch:02d}.h5',  # Save model with epoch number
        monitor='val_accuracy',            # Save based on validation accuracy
        save_best_only=True,          # Save best only
        mode='max',
        save_weights_only=False,       # Save the full model, not just the weights
        verbose=1
    )

    # Check if the base model was frozen or unfrozen
    if base_model.trainable:
        print("Base model layers are unfrozen. Fine-tuning the base model.")
        first_fit = False  # We need to fine-tune
    else:
        print("Base model layers are frozen. Training the new layers first.")
        first_fit = True  # We only train the new layers initially
    print('\n\nWARNING: \n Load optimizer state not implemented yet!!')#TODO

    #only do this first fit if you have forzen base layers: (unfreeze after that completes.)
    if first_fit:
        optimizer_state_saver = OptimizerStateSaver(storagefolder, 0)
        model.fit(
            train_gen, 
            steps_per_epoch=len(X_train) // batchsize,  # Number of batches per epoch
            epochs=max_epochs,
            validation_data=val_gen, 
            validation_steps=len(X_test) // batchsize,  # Number of validation batches per epoch
            callbacks=[lr_scheduler, early_stopping, model_checkpoint, optimizer_state_saver]
        )

        #the second .fit() method is with unfrozen base: this should be more precise for fine-tuning
        base_model.trainable = True
        model.compile(optimizer=AdamW(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print('compiled model with unfrozen layers;')
    #This is the second fit for unfrozen base layer. 
    optimizer_state_saver = OptimizerStateSaver(storagefolder, 1)

    model.fit(
        train_gen, 
        steps_per_epoch=len(X_train) // batchsize, 
        epochs=max_epochs,
        validation_data=val_gen,
        validation_steps=len(X_test) // batchsize,
        callbacks=[lr_scheduler, early_stopping, model_checkpoint, optimizer_state_saver]
    )    
    return model


def get_X_y(df, target, drop = []): 
    drop.append(target)
    X = df.drop(columns=drop)
    y = df[target]
    return [X, y]

def convert_ndarrays_to_lists(obj):
    """
        converts the inferred results with numpy arrays back to
        serializable JSON.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert ndarray to list
    elif isinstance(obj, dict):
        return {key: convert_ndarrays_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays_to_lists(item) for item in obj]
    else:
        return obj