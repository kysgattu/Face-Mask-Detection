# Face-Mask-Detection

In the recent times of the pandemic, first line protection against the contract of disease is using a face mask. When going out, it is essential to cover our face(nose and mouth) with a mask. The project developed is a Convolutional Neural Network which detects whether a person is wearing a face mask or not.

I have taken a comprehensive dataset 3 seta of images containing images of two classes Mask and No mask. I have trained the model using Keras Sequential model with Adam optimising by performing training by varying different attributes of images in about 30 Epochs.

I have saved the model as as MD5 model 'FaDetNet.h5'.

Testing is done by two ways i.e. using a video input and photo input.I have used Opencv library of python for integrating the web cam of the laptop to capture images and video which are tested with the prediction function of the trained model to get a binary output.In case of image, if more number of captured images are predicted as No Mask then output will be given as NO Mask else it will be given as Mask and in case of video, all the faces present in each frame are checked for masks using trained model and result is shown above the face and a Beep Sound is given if there is no mask.

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
        - [Testing using Image](#img-test)
            - [Capturing images](#img-capture)
            - [Testing and Implementing](#img-implem)
        - [Testing using Video](#testing)
            - [Capturing Video](#vid-capture)
            - [Testing and Implementing](#vid-implem)
    - [Test Results](#results)
 
- [Developers](#developers)
- [Links](#links)
- [References](#references)            

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


## System Modules <a name='modules'></a>

> ### Training the System <a name='training'></a>

#### Collection of picture Data <a name='data-collection'></a>
- The training set is an essential component of the network. We use dataset from a challenge in kaggle
- Dataset contains images of two defined classes 

#### Building and Training Model <a name='build'></a>

- We train the model using Keras sequential API of Tensorflow to train the model in various aspects and different angles and ranges of picture properties. We train the model in 30 epochs for gaining good accuracy. We save the model as FaDetNet.h5 for further testing and implementing of trained model.


> ### Testing the System <a name='testing'></a>

#### Testing using Image <a name='img-test'></a>

##### Capturing images <a name='img-capture'></a>
- We use methods in OpenCV library to capture images using the web camera of laptop and save the a set of 10 images in a rapid succession to test the model trained.
- These images are saved into a seperate folder 'input' 

##### Testing and Implementing Model <a name='img-implem'></a>

- The images recorded are tested using the predict method of the model trained earlier i.e., FaDetNet.h5. 
- All the images are tested to get output Mask or No Mask. If the number of Mask outputs are greater than No mask outputs, the Final output is given as Mask and vice versa.

#### Testing using Video <a name='vid-test'></a>

##### Capturing video <a name='vid-capture'></a>
- We use methods in OpenCV library to capture a video using the web camera of laptop and save the video to test the model trained.
- These video is saved into 'input' 

##### Testing and Implementing Model <a name='vid-implem'></a>

- The video recorded is divided into a definite number of frames and each frame is tested using the predict method of the model trained earlier i.e., FaDetNet.h5. 
- Then the result is shown by highlighting faces in each frame of the video as No Mask and Mask.
- If Mask is not detected on the face, A beep sound alert is given.

> ### Test Results <a name='results'></a>

![alt tag](https://github.com/kysgattu/Face-Mask-Detection/blob/main/Results/Mask.png)
![alt tag](https://github.com/kysgattu/Face-Mask-Detection/blob/main/Results/No%20Mask.png)
![alt tag](https://github.com/kysgattu/Face-Mask-Detection/blob/main/Results/PhotoTest.png)


## Developer <a name='developers'></a>
* Kamal Yeshodhar Shastry Gattu

## Links <a name='links'></a>

GitHub:     [G K Y SHASTRY](https://github.com/kysgattu)

Contact me:     <gkyshastry0502@gmail.com> , <kysgattu0502@gmail.com>

## References <a name='references'></a>

[[1] COVID Face Mask Detection Dataset
](https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset)



###### Note:
For Trained Model Run FaDetNet.ipnyb or FaDetNet.py or just send me an email ;-)
