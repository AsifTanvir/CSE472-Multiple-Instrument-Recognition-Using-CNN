# CSE472-Multiple-Instrument-Recognition-Using-CNN

## Introduction
In this project we have used Convolutional Neural Network for Multiple instrument recognition from audio files. The audio files have been collected from open data source called
Openmic, [which can be found here](https://zenodo.org/record/1432913#.X-eXW_kzZPa). The code is still ongoing as 
the model performs very poorly and we are trying to improve it.

## Preprocessing
For preprocessing we have used librosa. We have used MFCC features instead of time features. We extract mffcc feature from each audio snippet and store it as json in each separate
folders. The shape of the trainabale dataset is (20000, 431, 13) and the shape of the Labels is (20000, 20). We have 20 instruments and the label array is a boolean array of each 
songs specifying which instrument it has.

## Model
For training the data, we have used CNN model. We have used keras sequential models for our data. The model has:
1. 3 convolution layers with relu activation.
2. Each layer with Max Pooling with (3,3) pool size and Batch Normalization.
3. 2 Dense layers with flatten input.

For output layer we have used “Sigmoid” activation function. For optimizer we have used Adam with learning rate  = 0.001. The loss function is “binary-crossentropy”, as we have 
multi labels to predict.

