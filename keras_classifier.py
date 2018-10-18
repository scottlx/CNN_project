from google.colab import drive
drive.mount('/content/drive/')
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
import numpy as np
import pandas as pd
from google.colab import files


classifier = Sequential()
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
classifier.add(Convolution2D(64, (5, 5), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(activation="relu", units=512))
classifier.add(Dense(activation="sigmoid", units=1))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

training_set = train_datagen.flow_from_directory(
    '/content/drive/My Drive/flowers_training',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_set= test_datagen.flow_from_directory(
    '/content/drive/My Drive/flowers_test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

classifier.fit_generator(
    training_set,
    steps_per_epoch=56,
    epochs=10,
)


test_set.reset()
pred=classifier.predict_generator(test_set,verbose=1)
print(pred)

predictions = [round(x[0]) for x in pred]
print(predictions)
filenames=test_set.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)
files.download('results.csv')