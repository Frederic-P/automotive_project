# Flask Tagging. 

A minimum of manual tagging is needed to acquire training data for the binary tagging phase and the angle tagging phase. 

## Launching the Flask APP
Please ensure you've gone through the proper setup and set up the config file in the `config` directory. A username and password a valid countrycode should all be set. This guide assumes the data collecting phase and the boundingbox-phase have been completed or has been completed for at least one of the available countries. This also assumes that MYSQL is set up and running. 

The country has to be completed (reason for this is that it will randomly select a brand, then model, then listing, then image)... if it only gets trained on a single brand that is present in the countryfolder, it might miss important signals and missclassify on completely unseen brands. 

to start the app: 
- open a terminal window with the activated virtual environment navigate to the directory that holds the flask-app and type : `python app.py`
- Open the addres in your browser if it doesn't open up by itself.
- log in using the usercredentials set in in your config file. 
- start taggin. 

## Tagging consistently. 

To tag data consistently please follow these guidelines and familiarize yourself with these two terms: 

**trackwidth** the width of an axle; i.e. if you drive your car over snow, the trackwidth is the area of the car between the outer imprints. 

**wheelbase**: when looking to the side of a vehicle, the wheelbase is the zone between the hubcap of the front wheel and the hubcap of the rear wheel. 

visual guide: 
https://en.wikipedia.org/wiki/Wheelbase#/media/File:Wheelbase_and_Track.png 

- images taken INSIDE the car's trackwidth are tagged as FRONT or REAR.
- images taken INSIDE the car's wheelbase are tagged as LEFT or RIGHT. 
- images that fall outside the car's trackwidth and wheelbase are diagonal views are tagged as FRONTLEFT, FRONTRIGHT, REARRIGHT or REARLEFT. 

## To err is human
that's why you can undo a misslabeled tag by clicking the handy button labeled `Oopsie`, it'll remove the last entry for the table holding the manual tags. 

## Shortcut
- There are a lot of crappy images in there, to speed up tagging, you can press `c` on your keyboard to flag the image as such in stead of having to click the actual button. 

## After tagging 
Once you've tagged about 20.000 images you can start training the Binary usability classifier and the angle tagger (phase 1 and 2) in `notebooks/machine learning notebooks`

