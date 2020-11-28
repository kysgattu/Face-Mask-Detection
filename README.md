# Face-Mask-Detection

In the recent times of the pandemic, first line protection against the contract of disease is using a face mask. When going out, it is essential to cover our face(nose and mouth) with a mask. The project developed is a Convolutional Neural Network which detects whether a person is wearing a face mask or not.

I have taken a comprehensive dataset 3 seta of images containing images of two classes Mask and No mask. I have trained the model using Keras Sequential model with Adam optimising by performing training by varying different attributes of images in about 30 Epochs.

I have saved the model as as MD5 model 'FaDetNet.h5'.

I have used Opencv library of python for integrating the web cam of the laptop to capture image and save 10 continuous images in an interval of 1 to 3 seconds. and the images are tested with the prediction function of the trained model to get a binary output. If more number of captured images are predicted as No Mask then output will be given as NO Mask else it will be given as Mask.

## Table of contents

- [Prerequisites](#prerequisites)
    - [Environment](#environment)
    - [Technologies Used](#technologies-used)
    - [Dataset Description](#dataset-description)

- [System Modules](#modules)
    - [Training the System](#training)
        - [Collection of picture Data](#data-collection)
        - [Building and Training Model](#build)
    - [Testing the System](#testing)
        - [Capturing images](#img-capture)
        - [Tessting and Implementing](#implem)


## Prerequisites <a name='prerequisites'></a>

### Environment <a name='environment'></a>

1. Python 3 Environment (Ancaonda preferred)
2. Python modules required:NumPy,Pandas,Opencv2,Matplotlib, Scikit-learn, Keras
3. Web Browser

OR
- Any Python3 IDE installed with above modules.


### Technologies Used <a name='technologies-used'></a>

1. Anaconda Jupyter Notebook

### Dataset Description <a name='dataset-description'></a>

Dataset is taken from a data source from a Challenge on Kaggle. It contains 3 folders of images namesly Train,Test and Validate. Each contains pictures of 2 classes No mask and Mask. 
- [Developers](#developers)
- [Links](#links)
- [References](#references)


## System Modules <a name='modules'></a>

> ### Training the System <a name='training'></a>

#### Collection of picture Data <a name='data-collection'></a>
- The training set is an essential component of the network. We use dataset from a challenge in kaggle
- Dataset contains images of two defined classes 

#### Building and Training Model <a name='build'></a>

- We train the model using Keras sequential API of Tensorflow to train the model in various aspects and different angles and ranges of picture properties. We train the model in 30 epochs for gaining good accuracy. We save the model as FaDetNet.h5 for further testing and implementing of trained model.


> ### Testing the System <a name='testning'></a>

#### Capturing images <a name='img-capture'></a>
- We use methods in OpenCV library to capture images using the web camera of laptop and save the a set of 10 images in a rapid succession to test the model trained.
- These images are saved into a seperate folder 'input' 

#### Testing and Implementing Model <a name='implem'></a>

- The images recorded are tested using the predict method of the model trained earlier i.e., FaDetNet.h5. 
- All the images are tested to get output Mask or No Mask. If the number of Mask outputs are greater than No mask outputs, the Final output is given as Mask and vice versa.

## Developer <a name='developers'></a>
* Kamal Yeshodhar Shastry Gattu

## Links <a name='links'></a>

GitHub:     [G K Y SHASTRY](https://github.com/kysgattu)

Contact me:     <gkyshastry0502@gmail.com> , <kysgattu0502@gmail.com>

## References <a name='references'></a>

[[1] COVID Face Mask Detection Dataset
](https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset)


