# Using the best model
The best approach according to notebook 3.6 was a single CNN model that is trained on all angles at once. This model is implemented in `/notebooks/machine learning notebooks/4 - predict.ipynb`

This notebook requires the YOLO model to be present (same configuration settings as described in *boxing.md* is needed); and the full CNN. The good thing is that we do not an additional infer-step to predict the angle and do not need to load 8 models (one per angle). In stead we can get away with 1 brand prediction model on top of the YOLO model. 

You need the CNN models shared at https://drive.google.com/file/d/1DBYWXS06_VDnx9qXGRDiibwN9lHJFlaP and store them in /final models as desbired in the Brand training.md document.