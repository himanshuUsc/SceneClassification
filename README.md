# SceneClassification
LSUN 2016 Challenge , Scene Classification
This repo consists of files which were used in doing the Scene Classification task given in LSUN 2016.
Googlenet.lua contains the inception  module provided by google.Model weights are stored in the folder 'dump'.This model was trained on imagenet dataset which had 1000 classes.
example.lua contains how to use googlenet.lua for custom inputs and outputs.
train.lua contains the code for training the model for the required challenge .
I added a linear layer of size 10 (as the challenge required 10 output classes) and used softamax classifier for predicting the output.
I trained this last layer over 200 epochs and each epoch had 10000 iterations.
Accuracy obtained in 82%.
