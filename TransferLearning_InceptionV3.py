import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import numpy as np
#import matplotlib.pyplot as plt
import os
import cv2
#import pandas as pd
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
print(tf.__version__)


Datadir = 'DatasetVideos'

Categories = ['AbrirTampa','FecharTampa','TrocarCartucho']

NUM_CHANNEL = 3
IMG_SIZE = 160
training_data = []
training_Result = []

for Cat in Categories:
    path = Datadir + '/' + Cat + '/frames'
   
    for fold in os.listdir(path):
         for img in os.listdir(path + '/' + fold):
            print(img)
            img_array = cv2.imread((path + '/' + fold + '/' + img) , cv2.IMREAD_COLOR)
            new_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
          #  plt.imshow(new_img_array)
           # plt.show()
            new_img_array = new_img_array/255.0
            training_data.append(new_img_array)
            #hotEncodedResult
            
            training_Result.append(Categories.index(Cat))
    
    
print(np.shape(training_data))
print(np.shape(training_Result))

#Reshape list

training_data = np.reshape(training_data, ((np.shape(training_data))[0],IMG_SIZE, IMG_SIZE,NUM_CHANNEL))

x_train, x_test, y_train, y_test = train_test_split(training_data, training_Result, train_size=0.8, shuffle = True, random_state=0)

print(y_train)


IMG_SHAPE = (IMG_SIZE,IMG_SIZE, 3)

base_model = InceptionV3(input_shape = IMG_SHAPE, include_top = False, weights='imagenet')
base_model.trainable = False

#model = tf.keras.Sequential([base_model,keras.layers.GlobalAveragePooling2D(),Dense(1, activation='sigmoid')])
print(base_model.output)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(16, activation='relu')(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
    
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


#logdir

tensorboard = TensorBoard(log_dir="/data/tb")

#trainnig model  
FitModel = model.fit(x_train,y_train,epochs=5,batch_size=32,validation_data=(x_test, y_test), callbacks= [tensorboard])
    

#saving model
model.save('/data/my_model.h5') 
#validation

_, accuracy = model.evaluate(x_test,y_test)
print('Accuracy: %.2f' % (accuracy*100))

