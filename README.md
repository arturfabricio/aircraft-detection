# Study of New Approach for Aircraft Detection from Optical Satellite Imagery with Deep Learning

Student Project for DTU Course 02456 - Deep Learning

## General Structure ##

This repo consistents of two *src* folders containing the following elements: 

 ### *src* ###
* The full model and training process. The code can be tested running the **main.py** file.
* **train.ipynb** is a notebook to illustrate the training loop working as intended.
* **inference.ipynb** is a notebook which allows to test the results of the gathered weights from the training.  
 ### *src - a new hope* ###
* The model only with convolutional layers and training process for pretraining. The code can be tested running the **main.py** file.

The trained models with loss and the used hyperparameters will be logged in the **data** folder. Within the **main.py** it is possible to change the amount of images trained on, and the frequency for the logging of the models which can be used to validate that the code runs faster when not loading all of the images. 

## Dataset ##

The dataset can be found on the following website: https://www.cosmiqworks.org/rareplanes/ and can be downloaded following their instructions. Inside the RarePlanes folder you download, the training data is in "train/RarePlanes_train_PS-RGB_tiled.tar", and the annotation file for training is "instances_train_aircraft". For test, it's exactly the same paths, but just *test* instead of train. The notation files should be included in a folder called **annot** and the train and test images shold be included in seperate folders with the names **train** and **test**. The previous three folders should all be inside the **data** folder, created in the root of the repo. 

