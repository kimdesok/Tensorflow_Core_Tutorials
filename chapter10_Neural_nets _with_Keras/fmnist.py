import tensorflow as tf
import tensorflow_datasets as tfds
import utils

print(tf.__version__)

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Data loading using Keras

# Fashion Mnist
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print('Train image shape, ', train_images.shape)
print('Train image data type, ', train_images.dtype)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('Train image shape, ', train_images.shape)
print('No of labels, ', len(train_labels))
print('Train labels, ', train_labels)

print('Test image shape, ', test_images.shape)
print('No of labels, ', len(test_labels))
print('Test labels, ', test_labels)

#utils.show_fig(train_images[0])

# preparation of images
train_images = train_images / 255.0
test_images = test_images / 255.0

# prepare the validation set
val_set = 10000
validation_images, train_images = train_images[:val_set], train_images[val_set:]
validation_labels, train_labels = train_labels[:val_set], train_labels[val_set:]

#utils.check_images(train_images, class_names, train_labels)

# callbacks
model_name = "Adam.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_name,
    save_best_only=True
)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10)
])

print('Model layers,', model.layers)
print('Model layer [1]s name,', model.layers[2].name)
weights, biases = model.layers[2].get_weights()
print("Model layer [1]s weights, \n", weights)
print("Model layer [1]s weights shape, ", weights.shape)
print("Model layer [1]s biases, \n ", biases)
print("Model layer [1]s biases shape", biases.shape)

# Compile the model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
#tf.keras.utils.plot_model(model, "my_fashion_mnist_model.png", show_shapes=True)

# Train the model (or retrieve it if it has been fitted)
try:
    model = tf.keras.models.load_model(model_name)
except:
    history = model.fit(train_images, train_labels, validation_data=(validation_images, validation_labels), callbacks=[checkpoint], epochs=30)
    print("History parameters, ", history.params)
    print("History keys, ", history.history.keys())
    utils.plot_loss(history)

# Evaluate the model

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy, ', test_acc)

# make predictions

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
prediction_arrays = probability_model.predict(test_images)
'''
For testing a single image
img = (np.expand_dims(img.0)) # (1, 28, 28)
predictions_single = probability_model.predict(img)
'''

print(prediction_arrays[0])
print('Assigned label, ', class_names[test_labels[0]])
print('Predicted label, ', class_names[np.argmax(prediction_arrays[0])])

# Verify predictions

utils.check_labels(test_images, prediction_arrays, class_names, test_labels)
