#!/usr/bin/env python
# coding: utf-8

# In[1]:


#dataset: h

# !wget http://cb.lk/covid_19


# In[2]:


# !unzip covid_19


# In[3]:


train_path='mlops/CovidDataset/Train'
val_path ='mlops/CovidDataset/Val'


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image


# In[5]:


#CNN
model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu', input_shape=(224,224,3)))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))



model.add(Conv2D(128,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense (1, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])


# In[6]:


model.summary()


# In[7]:


#training

train_datagen=image.ImageDataGenerator(
   rescale=1./255,
   shear_range=0.2,
   zoom_range=0.2,
   horizontal_flip=True,
)

test_dataset=image.ImageDataGenerator(rescale=1./255)


# In[8]:


train_generator=train_datagen.flow_from_directory(
    'mlops/CovidDataset/Train',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)


# In[9]:


train_generator.class_indices


# In[10]:


validation_gen=test_dataset.flow_from_directory(
    'mlops/CovidDataset/Val',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'

)


# In[11]:


hist=model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=5,
    validation_data=validation_gen,
    validation_steps=2
)


# In[13]:


model.save('model_CD.h5')


# In[14]:


model.evaluate_generator(train_generator)


# In[15]:


model.evaluate_generator(validation_gen)


# # Testing Images

# In[16]:


mod=load_model('model_CD.h5')


# In[17]:


import os


# In[18]:


yact=[]
ytest=[]


# In[19]:


for i in os.listdir('mlops/CovidDataset/Val/Normal/'):
  img=image.load_img('mlops/CovidDataset/Val/Normal/'+i, target_size=(224,224))
  img=image.img_to_array(img)
  img=np.expand_dims(img,axis=0)
  p=mod.predict_classes(img)
  ytest.append(p[0,0])
  yact.append(1)


# In[20]:


for i in os.listdir('mlops/CovidDataset/Val/Covid/'):
  img=image.load_img('mlops/CovidDataset/Val/Covid/'+i, target_size=(224,224))
  img=image.img_to_array(img)
  img=np.expand_dims(img,axis=0)
  p=mod.predict_classes(img)
  ytest.append(p[0,0])
  yact.append(0)


# In[21]:


yact=np.array(yact)
ytest=np.array(ytest)


# In[22]:


from sklearn.metrics import confusion_matrix


# In[23]:


cm=confusion_matrix(yact,ytest)





