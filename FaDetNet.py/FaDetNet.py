#!/usr/bin/env python
# coding: utf-8

# # FACE MASK DETECTION
# 
# ### Training a Convolutional Neural network to detect whether the person an image or video captured is wearing a Face mask or not.
# - Dataset Contains 3 sets of images containing images of two classes for training the system

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os


# In[2]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


# In[3]:


data = 'dataset'
train = os.path.join(data,'Train')
test = os.path.join(data,'Test')
valid = os.path.join(data,'Validation')

train_mask = os.path.join(train,'Mask')
train_nomask = os.path.join(train,'Non Mask')


# In[4]:


train_mask_list = os.listdir(train_mask)
print(train_mask_list[:10])

train_nomask_list = os.listdir(train_nomask)
print(train_nomask_list[:10])


# In[5]:


import matplotlib.image as mpimg


# In[6]:


for i in train_mask_list[0:8]:
    print(i)


# In[7]:


mask_pic = []
for i in train_mask_list[0:8]:
  mask_pic.append(os.path.join(train_mask,i))

nomask_pic = []
for i in train_nomask_list[0:8]:
  nomask_pic.append(os.path.join(train_nomask,i))

print(mask_pic)
print(nomask_pic)

merged_pics = mask_pic+nomask_pic


# In[8]:


nrows = 4
ncols = 4
plt.figure(figsize=(12,12))

for i in range(0,len(merged_pics)):
  data = merged_pics[i].split('\\',2)[2]
  sp = plt.subplot(4,4,i+1)
  sp.axis('Off')
  image = mpimg.imread(merged_pics[i])
  sp.set_title(data,fontsize=10)
  plt.imshow(image,cmap='gray')

plt.show()


# In[9]:


train_datagen = ImageDataGenerator(rescale=1./255,
                                   zoom_range = 0.2,
                                   rotation_range = 40,
                                   horizontal_flip = True
                                   )
test_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train, target_size=(150,150),
                                                    batch_size = 32, class_mode = 'binary'
                                                    )
test_generator = test_datagen.flow_from_directory(test, target_size=(150,150),
                                                    batch_size = 32, class_mode = 'binary'
                                                    )
valid_generator = validation_datagen.flow_from_directory(valid, target_size=(150,150),
                                                    batch_size = 32, class_mode = 'binary'
                                                    )


# In[10]:


train_generator.class_indices


# In[11]:


model = Sequential()
model.add(Conv2D(32,(3,3),padding='SAME',activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(64,(3,3),padding='SAME',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.summary()


# In[12]:


model.compile(Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])


# In[13]:


history = model.fit(train_generator, epochs = 30, validation_data = valid_generator)


# In[14]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['trainig','validation'])
plt.title('Training and validation loss')
plt.xlabel('epoch')


# In[15]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['trainig','validation'])
plt.title('Training and validation accuracy')
plt.xlabel('epoch')


# In[17]:


model.save("FaDetNet.h5")


# In[ ]:


test_loss , test_acc = model.evaluate(test_generator)
print('test acc :{} test loss:{}'.format(test_acc,test_loss))

