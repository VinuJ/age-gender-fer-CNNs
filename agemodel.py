
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

cropped_faces = pathlib.Path('X:/PRBX/mycode/age+genderdataset')

batch_size = 32
height = 220
width = 220

faces = list(cropped_faces.glob('*.jpg'))

df = pd.DataFrame(columns = ['images', 'age'])

for face in faces:
    new_face = str(face)
    
    face = new_face.split("\\")
    
    face_name = face[4]
    face = face_name.split("_")
    
    face[0] = int(face[0])
    
    if face[0] < 21:
        face[0] = '0-20' 
    elif face[0] < 41:
        face[0] = '21-40'
    elif face[0] < 61:
        face[0] = '41-60'
    elif face[0] < 81:
        face[0] = '61-80'
    else:
        face[0] = '81-116' 
    
    temp_df = {'images': new_face, 'age': face[0]}
    df = df.append(temp_df, ignore_index = True)

# Splitting training and testing data
X_data = df.images
y_data = df.age
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2)

temp_df = {'images': X_train,'age': y_train}
df_train = pd.concat(temp_df , axis = 1)

# Splitting training and validation data
X_data = df_train.images
y_data = df_train.age
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size = 0.2)

# Training set
temp_df = {'images': X_train,'age': y_train}
train = pd.concat(temp_df, axis = 1)

# Validation set
temp_df = {'images': X_val,'age': y_val}
val = pd.concat(temp_df, axis = 1)

# Testing set
temp_df = {'images': X_test,'age': y_test}
test = pd.concat(temp_df, axis = 1)


# Data Augmentation #

training_set_generator = ImageDataGenerator(
    rotation_range = 45,
    horizontal_flip = True,
    #zca_whitening = True,
    rescale = 1/255
)

testing_set_generator = ImageDataGenerator(rescale = 1/255)

train_set = training_set_generator.flow_from_dataframe(
    dataframe = train,
    x_col = 'images',
    y_col = 'age',
    seed = 31,
    batch_size = batch_size,
    target_size = (height, width),
    class_mode ="sparse",
    shuffle = True
)

test_set = testing_set_generator.flow_from_dataframe(
    dataframe = test,
    x_col = 'images',
    y_col = 'age',
    batch_size = batch_size,
    target_size = (height, width),
    class_mode ="sparse",
    shuffle = True
)

val_set = training_set_generator.flow_from_dataframe(
    dataframe = val, 
    x_col = 'images',
    y_col = 'age',
    seed = 31,
    batch_size = batch_size,
    target_size = (height, width),
    class_mode = "sparse",
    shuffle = True
)

# Configuiring layers #

conv = layers.Conv2D(32, 3, activation='relu')
max_pool = layers.MaxPooling2D(pool_size = (2, 2))
norm = layers.experimental.preprocessing.Rescaling(1/255)
drop = layers.Dropout(0.2)


callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', patience = 5)

age_model = tf.keras.Sequential([
    
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
    
  layers.Flatten(),
  layers.Dense(128, activation = 'relu'),
  
  layers.Dense(5)

])

# Compiling model #

age_model.compile(
  loss = tf.losses.SparseCategoricalCrossentropy(from_logits = True),
  optimizer = 'adam',
  #optimizer = 'SGD',
  metrics = ['accuracy']
  )

# Fitting model #

history = age_model.fit(
    train_set,
    epochs = 1,
    validation_data = val_set,
    callbacks = callback,
    shuffle = False
)

# Model summary #
age_model.summary()

# Evaluating model on testing set

results = age_model.evaluate(test_set)

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
