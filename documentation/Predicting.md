# Using the best model

Congratulations if you read this far, now imagine typing it. Anyhow. 
If you want to use the models and see how they'd be implemented in a an actual prediction product, you can use the code as it is implemented in `/notebooks/machine learning notebooks/4 - predict.ipynb`. The implementation here was done base on what we saw in notebook 3.6 In essence we learned that cropping is useful; that CNN models perform better on the dataset than the ViT model and that it is better to use one big model versus the eight angle-specific ones. (this has the added advantage that we don't lose time trying to infer the angle)

This notebook requires the YOLO model to be present (same configuration settings as described in *boxing.md* is needed); and the full CNN. The good thing is that we do not an additional infer-step to predict the angle and do not need to load 8 models (one per angle). In stead we can get away with 1 brand prediction model on top of the YOLO model. 

You need the CNN models shared at https://drive.google.com/file/d/1DBYWXS06_VDnx9qXGRDiibwN9lHJFlaP and store them in /final models as desbired in the Brand training.md document. After that you just press play and let it go. The brand prediction model will give a predicition to an image using of the thirty classes it was trained on.