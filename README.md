# Fake-Image-Detection
A model that distinguishes between real images and GAN generated fake images.  

Dataset: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images  

Dataset consists of two sub-directories: train and test. Both sub-directories have 2 classes: Fake and Real. Test sub-directory has 10,000 real images and 10,000 fake images. Train sub-directory has 50,000 real images and 50,000 fake images.

I have used a simple and straight-forward CNN model for this project.
## Pre-Processing:
Image size is 32x32. Batch size is 32.
We first load the data using tensorflow from the directory as train and validation datasets. No other pre-processing is done as the Kaggle dataset itself is pre-processed.  

## Model Architecture and Training:
The CNN model has 4 layers that is then connected to an ANN that has 2 layers to predict the output.  
The CNN model consists of 1 Rescaling Layer, 1 Convolutional layer, 1 Max Pooling Layer and 1 Flatten layer.  
The ANN model consists of 1 dense layer and 1 output sigmoid dense layer.  
The model is compiled using Adam Optimizer and Binary Cross-entropy loss. The model is then trained on the loaded dataset.  
The model has a validation accuracy of 94% at the end of 5 epochs.  

![image](https://github.com/Akshath-0406/Fake-Image-Detection/assets/96140050/66a9754b-cf8b-448a-a05e-89f55c20eed5)  

![image](https://github.com/Akshath-0406/Fake-Image-Detection/assets/96140050/560265cf-2703-4b40-8aa8-d9a359a5deb3)
