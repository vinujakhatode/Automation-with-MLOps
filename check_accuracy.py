import numpy as np
import keras
from keras.models import *
from keras.preprocessing import image
from sklearn.metrics import accuracy_score


mod=load_model('model_CD.h5')

import os

yact=[]
ytest=[]


for i in os.listdir('mlops/CovidDataset/Val/Normal/'):
  img=image.load_img('mlops/CovidDataset/Val/Normal/'+i, target_size=(224,224))
  img=image.img_to_array(img)
  img=np.expand_dims(img,axis=0)
  p=mod.predict_classes(img)
  ytest.append(p[0,0])
  yact.append(1)


for i in os.listdir('mlops/CovidDataset/Val/Covid/'):
  img=image.load_img('mlops/CovidDataset/Val/Covid/'+i, target_size=(224,224))
  img=image.img_to_array(img)
  img=np.expand_dims(img,axis=0)
  p=mod.predict_classes(img)
  ytest.append(p[0,0])
  yact.append(0)


yact=np.array(yact)
ytest=np.array(ytest)


accuracy=accuracy_score(ytest,yact,normalize=True)

accuracy=accuracy*100
accuracy=int(accuracy)
print(accuracy)
