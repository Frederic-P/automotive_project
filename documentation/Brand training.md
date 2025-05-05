# Brand training

This section of the documentation describes how to use all notebooks in `notebooks/machine learning notebooks/` that have a name starting with '3.'. The naming convention of these notebooks is 3.x - title, where x is the execution order of the notebooks to be followed; the title is a desicription at glance of the goal if the notebook. 

training of the CNNS was done using google Colab. As such notebook 3.3 and 3.4 are written for a COLAB environment in mind. To move the data to COLAB, you have to ZIP your data using the notebook `notebookes/to_colab.ipynb`

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
#TODO

## 3.4 - Large_model_training(COLAB Version),ipynb
#TODO

## 3.5 - Brand classification - vision transformer (angled).ipynb
#TODO

## 3.5 - Bis - TINY ViT comparison.ipynb
#TODO

## 3.6 - Brand performance comparison.ipynb
#TODO
