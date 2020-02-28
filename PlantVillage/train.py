# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:54:04 2019

@author: Asus
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


model=Sequential()

model.add(Conv2D(32,3,3,input_shape=(128,128,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, 3, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))


model.add(Flatten()) #it converts multidimension to one dimension
model.add(Dense(output_dim = 128, activation = 'relu',init='random_uniform'))
model.add(Dense(output_dim = 1, activation = 'sigmoid',init='random_uniform'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('',
                                                 target_size = (128, 128),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('',
                                            target_size = (128, 128),
                                            batch_size = 10,
                                            class_mode = 'categorical')
print(training_set.class_indices)

model.fit_generator(training_set,
                         samples_per_epoch = ,
                         epochs = 15,
                         validation_data = test_set,
                         nb_val_samples = )

model.save('internship.h5') 