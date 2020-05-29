
train_path='mlops/CovidDataset/Train'
val_path ='mlops/CovidDataset/Val'

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image


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

model.summary()

#training

train_datagen=image.ImageDataGenerator(
   rescale=1./255,
   shear_range=0.2,
   zoom_range=0.2,
   horizontal_flip=True,
)

test_dataset=image.ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    'mlops/CovidDataset/Train',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)

train_generator.class_indices

validation_gen=test_dataset.flow_from_directory(
    'mlops/CovidDataset/Val',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'

)
epoch=5
hist=model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=epoch,
    validation_data=validation_gen,
    validation_steps=2
)

model.save('model_CD.h5')

model.evaluate_generator(train_generator)

model.evaluate_generator(validation_gen)
