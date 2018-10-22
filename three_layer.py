from google.colab import drive
drive.mount('/content/drive/')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from keras import backend as K
import numpy as np
import pandas as pd
from google.colab import files
import os
import tensorflow as tf


# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = '/content/drive/My Drive/flowers/train'
validation_data_dir = '/content/drive/My Drive/flowers/validation'
nb_train_samples = 1616
nb_validation_samples = 200
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']

classifier = Sequential()
classifier.add(Convolution2D(32, (3, 3),input_shape = input_shape, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(64, (3, 3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Flatten())
classifier.add(Dense(activation="relu", units=64))
classifier.add(Dropout(0.5))
classifier.add(Dense(activation="sigmoid", units=1))
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

tpu_model = tf.contrib.tpu.keras_to_tpu_model(
classifier,
strategy=tf.contrib.tpu.TPUDistributionStrategy(
    tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator= test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

classifier.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)
test_set= test_datagen.flow_from_directory(
    '/content/drive/My Drive/flowers_test',
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='binary',
    shuffle =False
)
test_set.reset()
pred=classifier.predict_generator(test_set,verbose=1, steps = 1)
print(pred)

predictions = [round(x[0]) for x in pred]
print(predictions)
filenames=test_set.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)
files.download('results.csv')