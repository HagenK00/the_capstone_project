import tensorflow as tf
import numpy as np
import os
import glob
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

cropped_images_red = glob.glob('cropped_sorted_training/red/*.jpg')
cropped_images_yellow = glob.glob('cropped_sorted_training/yellow/*.jpg')
cropped_images_green = glob.glob('cropped_sorted_training/green/*.jpg')
cropped_images_false = glob.glob('cropped_sorted_training/false/*.jpg')

labels = []
x_data = []

for img in cropped_images_red:
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(64,64))
    x_data.append(image)
    labels.append(0)

for img in cropped_images_green:
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(64,64))
    x_data.append(image)
    labels.append(2)

for img in cropped_images_yellow:
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(64,64))
    x_data.append(image)
    labels.append(1)

for img in cropped_images_false:
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(64,64))
    x_data.append(image)
    labels.append(4)
x_data = np.asarray(x_data)
one_hot_labels = to_categorical(labels, num_classes=5)

x_train, label_train = shuffle(x_data, one_hot_labels)

model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(64,64,3)))
model.add(Conv2D(6,(5,5),strides=(2, 2), padding='same',activation='relu',use_bias=True,kernel_initializer='TruncatedNormal', bias_initializer='zeros'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
model.add(Conv2D(12,(3,3),strides=(2, 2), padding='same',activation='relu',use_bias=True,kernel_initializer='TruncatedNormal', bias_initializer='zeros'))

model.add(Dropout(0.25))

model.add(Conv2D(18,(1,1),strides=(1, 1), padding='valid',activation='relu',use_bias=True,kernel_initializer='TruncatedNormal', bias_initializer='zeros'))
model.add(Flatten())
model.add(Dense(100,activation='relu',kernel_initializer='TruncatedNormal',bias_initializer='zeros'))

model.add(Dropout(0.5))

model.add(Dense(10,activation='relu',kernel_initializer='TruncatedNormal',bias_initializer='zeros'))

model.add(Dense(5,activation='relu',kernel_initializer='TruncatedNormal',bias_initializer='zeros'))

optimizer = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

model.fit(x=x_train, y=label_train, epochs=5, batch_size=3)
model.save('model_.h5')

