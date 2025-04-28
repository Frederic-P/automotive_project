# Bounding box applicator. 

A pre-made model is used to apply bounding boxes to images; this assumes data is downloaded from Autoscout and is available on the computer; The Yolo-box taggers should be run on the same computer that is hosting the images (FTP introduces a significant overhead and therefore was not integrated in the project).

## Motivation: 
A subquestion of this project is whether or not CNN models perform better if they are forced to learn on a part of the image versus the whole image. That's why this component is included. 

## Running bounding boxes:
Bounding boxes can be ran automatically by setting up a CRON-service (not covered in this guide) or can be set up by triggering the boxtagger script manually. The script responsible for bounding box tagging is `automated_car_detector.py` stored in the root directory of this repository. A known Linux issue needs to be bypassed to make it work (see: https://askubuntu.com/questions/1299255/how-can-i-solve-no-module-named-lzma). 

If you want to visually see what and how the bounding boxes are applied, you can run the notebook at `/experimental code/detect_car.ipynb`. This notebook will iterate over the image directory, randomly select a brand, model and listing and generate bounding boxes for the car. Bounding box detection is not free of mistakes. 

## Storage of bounding boxes: 
BBOX data is stored in the MYSQL-server and is being used in various phases of the end-to-end project (binary classifier, angle classifier, brand classifier). The returned coordinates by YOLO are alwyas the TOPLEFT and BOTTOMRIGHT; a certainty score is stored too. 

## Concurrent phases: 
You are allowed to use the next phase (FLASK-tagging) while the bounding boxes are being calculated. 