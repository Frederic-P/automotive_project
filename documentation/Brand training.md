# Brand training

This section of the documentation describes how to use all notebooks in `notebooks/machine learning notebooks/` that have a name starting with '3.'. The naming convention of these notebooks is 3.x - title, where x is the execution order of the notebooks to be followed; the title is a desicription at glance of the goal if the notebook. 

training of the CNNS was done using google Colab. As such notebook 3.3 and 3.4 are written for a COLAB environment in mind. To move the data to COLAB, you have to ZIP your data using the notebook `notebooks/to_colab.ipynb`. This notebook will crop your images to the bounding box to reduce disk space required on Google Drive. It'll use the images geneated in step 3.1 (explained below)

The correct execution order of all 3.x notebooks is the order of the h2-headers in this markdown file. 

This phase can only be started if the following conditions are met: 
- all data is collected
- all images have a BBOX
- all images have the four usability scores applied to them
- all images have an angle tag

## 3.1 - Brand classifier.ipynb

This notebook fetches all data from the database that meets the usability treshold. Performs a train test validation split (70-20-10)
- the validation data is not touched until subphase 3.6. Images that are part of this are NOT augmented and are written to a CSV file. 
- the testdata is used during training in subphases 3.2, 3.3, 3.4 and 3.5 as X_tes tand y_test. These are written to CSV file for reuse and consistency. 
- the traindata is augmented where needed so that the following conditions are met: Each of the 30 brands has 10.000 images for each of the 8 angles; this results in a dataset of 2.4 million images. Augmented data is stored on disk and saved in a CSV with absolute paths to either an augmented file or an original file. (There are no augmentations being made of data that is in the validation or test set). augmentation only happens on the 70% part! For pictures of the left and right angle we saw that these are the least frequent in the manually tagged images AND in the automatically tagged images. A first augmentation trick that was applied was to mirror images of the LEFT view so they became a RIGHT view and vice versa. This is still a more realistic augmenation than color jitter or severe gaussian noise. 

## 3.2 - TINY Brand classification - transfer learning (angled)
This notebook's goal is to decide if cropping an image is beneficial for a model's performance. We do this for five brands; we pick a brand with the least amount of augmentations (Volkswagen) and the brand with the most amount of augmentations (Alpine) and pick three more brands that have are evenly spaced in between automatically (ford, seat, suzuki). For these five brands the aim is to predict the brand using transfer learning techniques (RESNET50). This experiment found there is only to be gained from cropping an image (just as with angle tagging). With this in mind all other notebooks in phase 3 will now apply cropping logic. 

## 3.3 - 3.3 - Brand classification - transfer learning (angled - COLAB version).ipynb
Training was done in COLAB you need to use the to_colab notebook to get this phase started.!!
Phase 3.3 attemtps to classify all thirty brands of cars based on a trainset of 300.000 images for each of eight specific angles. A COLAB-version is available for this notebook. To use the COLAB notebook for training, you need to create a ZIP file of training and testdata for a specific angle; upload that to your Google Drive. The zip folders will be named *anglename*.zip (e.g. front.zip). The Zipfiles should be place in two folders from the root of your Google drive 'testdata' and 'traindata'. To train, the notebook needs access to your Google Drive. In the COLAB version it is CRITICAL that you set the ``crop``-constant to FALSE. Because cropping was already done by the to_colab notebook! In the non-colab version you can choose, just remember that cropping worked the best (see phase 3.2)


## 3.4 - Large_model_training(COLAB Version),ipynb
Phase 3.4 only has a COLAB version available; here CROPPING should be set to false as it uses the data from the to_colab notebook. You also need to upload ALL your zip files to Google drive, this will exceed the 15GB limit in Google Drive, you need to upload these zip files to other accounts, share them with you and make shortcuts in the main google drive in a subdirectory called 'shared_data'; this folder should now have shortcuts to other google drive accounts with the shortcuts being named 'traindata' and 'testdata', these names can be re-used. Insinde each of the shortcuts one or more zipfiles lives with a name that represents the kind of data it holds e.g. (front.zip). The google drive account you use to train should NOT hold zip files as you need the space there to store checkpoints - training will need multiple days, the checkpoints allow restarting once the 24hr allotment of compute time runs out. In this phase you re-use the same zip files as in step 3.3 but you train them all at once. The idea is to compare in a later phase what works better: training 8 models on specific angles; or train one big model. 

## 3.5 - Brand classification - vision transformer (angled).ipynb
Phase 3.5 explores the use of Vision Transformers. These models can be configured using `/config vit_config.json` as k-v pairs. 

A key is a unique name in the JSON file that represents a training profile. Follow these conventions for key names: 
- alwas start a keyname with ViTnn where n is a digit. 
- Models that are trained on a subset of brands should not have any suffixes added. 
- Models that are trained on all brands, but are stopped at EPOCH 4 have a suffix: `_max4`
- Models that are trained on all brands and for as long as needed have a suffix: `_FULL`

In your valuedict for each profile you can experiment with different settings for each profile: 
EXAMPLE:
```
      "SHAPE": 220,
      "PATCH_SIZE": 22,
      "BATCH_SIZE": 32,
      "LEARNING_RATE": 0.0005,
      "WEIGHT_DECAY": 0.0001,
      "TRANSFORMER_LAYERS": 8,
      "VECTORSIZE": 64,
      "HEADS": 4,
      "PATCHES": 100, 
      "TINY":false,
      "MAX_EPOCHS":100,
      "MLP_X": 2048, 
      "MLP_Y": 1024
```
These three parameters are critical:
- shape = x and y dimension of the image
- patchsize = x and y dimension of each patch (SHAPE should be a multiple of PATCH_SIZE)
- patches = (SHAPE/PATCH_SIZE)**2 (Explicitly give this)

- To limit a model to 4 epochs for `MAX_4` profiles set the `MAX_EPOCHS` key to 4
- To limit a model to a preselected subset of brands, set TINY to true (casesensitive)!!

The way this JSON driven config works is: it checks if a configuration has already run, skips it if it already did or trains according to the profile.


## 3.5 - Bis - TINY ViT comparison.ipynb
If you've completed a few training rounds of the configured Vision Transformers, you use notebook 3.5 bis to assess the performance of the ViT Models.  


## 3.6 - Brand performance comparison.ipynb
After training you should download the models you trained using Google Colab and copy them into the folder `/final models/` into the appropriate folder. There is an important folder hierarchy to keep in mind: 

```
    final models
    |____angles
    |____brand
            |____CNN_based_models
            .       |____front
            .       |____front_balanced
            .       |____frontleft
            .       |____frontright
            .       |____full
            .       |____left
            .       |____rear
            .       |____rearleft
            .       |____rearright
            .       |____right
            |____ViT_models
                    |____front        
```
- The model you got to predict the angle of a picture, you store in angles together with it's JSON file.
- CNN based models to predict the brand you should store in `brand/CNN_based_models` in the appropriately named folder. 
- the ViT_models are store in `brand/ViT_models` also in the appropriately named folder. 

The full set of models can be downloaded from: https://drive.google.com/file/d/1DBYWXS06_VDnx9qXGRDiibwN9lHJFlaP 
this excludes the binary taggers as they are not part of the infer-pipeline (phase4); it DOES include the angle model, even if it was proven to be unnecessary to use for inferring.
You run this notebook once you have one fully functional ViT model, eight angle-specific brand models using CNN and one non-angle specific CNN model. These models and their JSON dicts should be stored in the final_models dir. You run this notebook and look at all the fancy graphs that show up on your screen. After you're done, you sit back and relax. The ViT model does not come with a JSON file to decode labels, it uses the same labels as the CNNs