"""
Collection of classes for the Vision transformer; code is based 
on the KERAS tutorial for Vision Transformers. 

"""
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers, ops
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import json


def get_vit_configurations(file_path):     
    with open(file_path, 'r', encoding='utf-8') as conf: 
        return json.load(conf)
    

def evaluate_vit(configs, df, dir, apply_crop = True, batchsize = 256 ):
    results = {}
    actuals = {}

    # Run evaluation loop on all ids in the dict
    for vit_id, variant in configs.items():
        patch_size = variant["PATCH_SIZE"]
        shape = variant["SHAPE"]
        PATCHES = (shape // patch_size) ** 2
        mode = variant['TINY'] == True
        print(f"Evaluating ViT: {vit_id} with PATCH_SIZE={patch_size}, PATCHES={PATCHES}, Tinymode={mode}")
        #When in testmode, this is the hardcoded list of brands we'll use to srhink the dataset (speed up testing of notebooks.)
        if mode:
            brands = ['ford', 'opel', 'bmw', 'renault']
            validation_df = df[df['brand'].isin(brands)].copy()

        else:
            validation_df = df.copy()


        brands = validation_df.brand.unique()
        label_encoder = LabelEncoder()
        label_encoder.fit(brands)
        validation_df['y_encoded'] = label_encoder.transform(validation_df['brand'])
        angles = list(validation_df.model_label.unique())

        for angle in angles:
            name = f'{vit_id}-model_cropped={apply_crop}_angle={angle}.keras'
            model_path = os.path.join(dir, name)
            if not os.path.exists(model_path):
                print(f"Model {name} not found, skipping.")
                continue

            print(f"Loading model {name}")
            model = keras.models.load_model(model_path,
                                            custom_objects={"Patches": Patches,
                                                            "PatchEncoder": PatchEncoder})

            view_by_angle_test = validation_df.query('model_label==@angle').reset_index()
            generator = ImageDataGeneratorFromDF(
                df=view_by_angle_test,
                image_column='abs_path',
                label_column='y_encoded',
                batch_size=batchsize,
                target_size=(shape, shape),
                shuffle=False,
                apply_crop=apply_crop
            )

            predictions = []
            #caveat of Keras, it requires all batches to be of equal length; 
            # this is a way around the issue without the risk of accidentally padding
            # the final batch with a underrepresented test class. (adding 3 entries of a 
            # class that only occurs 4 times naturally)
            for x_batch, _ in generator:
                if len(x_batch)  == 0:
                    break
                batch_preds = model.predict_on_batch(x_batch)
                predictions.extend(batch_preds)
            results[vit_id] = predictions
            actuals[vit_id] = {
                'data': view_by_angle_test['y_encoded'],
                'encoder' : label_encoder
            }
    return [results, actuals]


class ImageDataGeneratorFromDF(Sequence):
    def __init__(self, df, image_column, label_column, batch_size, target_size=(224, 224), shuffle=True, apply_crop=False):
        self.df = df
        self.image_column = image_column
        self.label_column = label_column
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.apply_crop = apply_crop
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Select the batch of indexes
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Retrieve the batch of image paths and labels (ensure the data is aligned)
        batch_data = self.df.iloc[batch_indexes]  # This selects the correct rows for images and labels
        batch_image_paths = batch_data[self.image_column].values
        batch_labels = batch_data[self.label_column].values
        # batch_brands = batch_data['brand'].values

        # Debugging werid stuff keeps happening.
        # print(batch_labels, batch_brands)
        
        # Load and preprocess the images, applying crop if needed
        images = np.array([self.load_and_preprocess_image(i, image_path) for i, image_path in zip(batch_indexes, batch_image_paths)])
        
        return images, np.array(batch_labels)
    
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        # print('SHUFFLE DATA')
        if self.shuffle:
            np.random.shuffle(self.indexes)

        
    def load_and_preprocess_image(self, global_index, image_path):
        """Load image, apply cropping (if needed), and preprocess"""
        image = Image.open(image_path)
        
        # Apply crop if required (remember that -1 is the sentinel for NO CROP CONSTRAINT BY YOLO)
        if self.apply_crop and int(self.df.iloc[global_index]['yolobox_top_left_x']) > -1:
            # Get the crop coordinates for the image at the global index
            top_left_x = self.df.iloc[global_index]['yolobox_top_left_x']
            top_left_y = self.df.iloc[global_index]['yolobox_top_left_y']
            bottom_right_x = self.df.iloc[global_index]['yolobox_bottom_right_x']
            bottom_right_y = self.df.iloc[global_index]['yolobox_bottom_right_y']
            
            # Crop the image using the coordinates from the DataFrame
            image = image.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
        # Resize the image to the target size
        image = image.resize(self.target_size)
        
        # Convert image to numpy array and normalize if needed
        image = np.array(image) / 255.0  # Normalizing to [0, 1]
        
        return image
    


    # Implement the patch encoding layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim  
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches, 
            "projection_dim": self.projection_dim
            })
        return config
    


# Implement patch creation as a layer
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = tf.image.extract_patches(
            images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config