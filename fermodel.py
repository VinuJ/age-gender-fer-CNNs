# Standard imports #

import pandas as pd
import numpy as np
import tensorflow as tf
import pathlib

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Pre-processing #

fer_faces = pathlib.Path('X:/PRBX/mycode/ferdataset')

batch_size = 32
height = 220
width = 220

df = pd.DataFrame(columns=['images','emotion'])

emotion_classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

for label in emotion_classes:
    for face in list(fer_faces.glob(label+'/*.png')):
        temp_df = {'images':str(face), 'emotion':label}
        df = df.append(temp_df, ignore_index = True)

# Randomise dataframe order

df = df.sample(frac=1)

# Splitting training and testing data

X_data = df.images
y_data = df.emotion
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2)

temp_df = {'images': X_train, 'emotion': y_train}
df_train = pd.concat(temp_df, axis = 1)

# Splitting training and validation data
X_data = df_train.images
y_data = df_train.emotion
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size = 0.2)

# Training set
temp_df = {'images': X_train, 'emotion': y_train}
train = pd.concat(temp_df, axis = 1)


# Validation set
temp_df = {'images': X_val, 'emotion': y_val}
val = pd.concat(temp_df, axis = 1)


# Testing set
temp_df = {'images': X_test, 'emotion': y_test}
test = pd.concat(temp_df, axis = 1)

# Data Augmentation #

training_set_generator = ImageDataGenerator(
    rotation_range = 45,
    horizontal_flip = True,
    #zca_whitening = True,
    rescale=1/255
)

testing_set_generator = ImageDataGenerator(rescale = 1/255)

train_set = training_set_generator.flow_from_dataframe(
    dataframe = train,
    x_col = 'images',
    y_col = 'emotion',
    seed = 31,
    batch_size = batch_size,
    target_size = (height, width),
    class_mode ="sparse",
    shuffle = True
)

test_set = testing_set_generator.flow_from_dataframe(
    dataframe = test,
    x_col = 'images',
    y_col = 'emotion',
    batch_size = batch_size,
    target_size = (height, width),
    class_mode ="sparse",
    shuffle = True
)

val_set = training_set_generator.flow_from_dataframe(
    dataframe = val, 
    x_col = 'images',
    y_col = 'emotion',
    seed = 31,
    batch_size = batch_size,
    target_size = (height, width),
    class_mode = "sparse",
    shuffle = True
)

# Configuiring layers #

conv = layers.Conv2D(64, 3, activation = 'relu')
max_pool = layers.MaxPooling2D(pool_size = (2, 2))
norm = layers.experimental.preprocessing.Rescaling(1/255)
drop = layers.Dropout(0.2)


callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', patience = 5)

fer_model = tf.keras.Sequential([
  
  norm,
  conv,
  drop,
  max_pool,

  conv,
  drop,
  max_pool,

  conv,
  drop,
  max_pool,

  conv,
  drop,
  max_pool,

  conv,
  drop,
  max_pool,

  conv,
  drop,
  max_pool,

  conv,
  drop,
  max_pool,

  layers.Flatten(),
  layers.Dense(128, activation = 'relu'),
  
  layers.Dense(7)
  
])

# Compiling model #

fer_model.compile(
  optimizer = 'adam',
  #optimizer = 'SGD',
  loss = tf.losses.SparseCategoricalCrossentropy(from_logits = True),
  metrics = ['accuracy']
)

# Fitting model #

history = fer_model.fit(
    train_set,
    validation_data = val_set,
    epochs = 50,
    callbacks = callback,
    shuffle = False
)

# Model summary #
fer_model.summary()


# Evaluating model on testing set

results = fer_model.evaluate(test_set)

print("Testing loss: ", results[0])
print("Testing accuracy: ", results[1])


# Plotting model history #

used_epochs = len(history.history['loss'])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(used_epochs)
plt.figure(figsize = (10, 10))

# Plotting training and validation accuracy graph

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label = 'Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'upper left')
plt.title(
    'Training and Validation Accuracy',
    fontdict = {'fontsize': '15', 'color': 'black'}
)

# Plotting training and validation loss graph

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.title(
    'Training and Validation Loss', 
    fontdict = {'fontsize': '15', 'color': 'black'}
)

plt.show()
