#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


# In[3]:


from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img , img_to_array


# In[4]:


model=load_model('FaDetNet.h5')


# In[5]:


camera = cv2.VideoCapture(0)
mask=0
nomask=0
for i in range(10):
    return_value, image = camera.read()
    cv2.imwrite(os.path.join('input/' , 'opencv'+str(i)+'.png'), image)
    #cv2.imwrite(os.path.join('input/' , 'img.{}.jpg'.format(n)),transfer)
    #img_path='/content/'+fname
    img = load_img('input/opencv%d.png'%(i) , target_size=(150,150))
    images = img_to_array(img)
    images=np.expand_dims(images,axis=0)
    prediction = model.predict(images)
    if prediction==0:
        mask=mask+1
        print('m')
    else:
        nomask=nomask+1
        print('n')
if mask>nomask:
    print('MASK ON')
else:
    print('MASK OFF')

del(camera)


# In[ ]:




