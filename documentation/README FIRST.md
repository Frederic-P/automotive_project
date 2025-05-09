# Order of readme instructions. 

This project comes with multiple instruction files bundled in the documentation folder; each readme file is responsible for setting up a part of the project. 

The order to read the instructions is; the read order of the instructions is critical as following the correct sequence ensures that each component is set up properly before the next one is introduced. e.g. The output of boundin gboxes is rendered in the Flask Tagger and vital for all components that follow. Not completing the BBOX-calculations will lead to errors in subsequent steps; these errors are considered user error and are not handled. 



1) installation.md
- This notebook guides you through setting up the project environment and tells you which services are required to add on top. 
2) Data collecting.md
- Describes how images are collected and stored in the system for analysis.
3) Boxing.md
- Describes how bounding boxes are calculated for images.
4) FlaskTagging.md
- Describes how the tagging system works and how to use it.
5) Binary training.md
- Describes how the binary classification model is trained and how to use it.
- Describes how to use the binary models to determine the usability of each image in your downloaded set.
6) Angle training.md
- Describes how the angle classification model is trained and how to use it.
- Describes how to use the angle classifier to determin the angle of each image in your downloaded set; 
7) Brand training.md
- Describes how the brand classification model is trained and how to use it; describes the usage of all 3.x notebooks.
8) Predicting.md
- The brand training notebook shows that a non-angle specific trained approach using CNNs works the best. This model is implemented in a demoable prediction notebook that downloads a bunch of files locally, infers them, and displays each image with brand label prediction and certainty.