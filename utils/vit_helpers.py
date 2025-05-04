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

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def create_vit_classifier(input_shape, patchsize, imshape, output_classes, heads, vectorsize, transformer_layers, transfo_units, mlphead_units ):
    inputs = keras.Input(shape=input_shape)
    # Create patches.
    patches = Patches(patchsize)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder((imshape // patchsize) ** 2, vectorsize)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=heads, key_dim=vectorsize, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transfo_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, VECTORSIZE] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlphead_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(output_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model



def run_experiment_with_generators(model, filename, dir, lr, wd, traingen, valgen, maxepochs):

        # Path to the checkpoint
    checkpoint_dest_dir = os.path.join(dir, 'checkpoints')
    os.makedirs(checkpoint_dest_dir, exist_ok=True)
    checkpoint_name = filename.replace('.keras', '.weights.h5')  # otherwise you get an error
    checkpoint_filepath = os.path.join(checkpoint_dest_dir, checkpoint_name)

    # Resume from checkpoint if it exists
    if os.path.exists(checkpoint_filepath):
        print(f"Resuming training from checkpoint: {checkpoint_filepath}")
        model.load_weights(checkpoint_filepath)
    else:
        print("No checkpoint found. Starting training from scratch.")

    optimizer = keras.optimizers.AdamW(
        learning_rate=lr, weight_decay=wd
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )


    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    # ReduceLROnPlateau callback: same as for CNN learners. 
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # EarlyStopping callback
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True,
        verbose=1
    )

    # Fit the model using generators
    history = model.fit(
        traingen,
        epochs=maxepochs,
        validation_data=valgen,
        callbacks=[checkpoint_callback, reduce_lr_callback, early_stopping_callback],
    )

    # Load the best model weights
    model.load_weights(checkpoint_filepath)

    # Final evaluation
    _, accuracy, top_5_accuracy = model.evaluate(valgen)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    
    #save the model:
    model.save(os.path.join(dir,filename))
    return history



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
        