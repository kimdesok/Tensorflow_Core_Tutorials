import os
import numpy as np

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def plot_loss(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    save_fig("keras_learning_curves_plot")
    plt.show()
    plt.close()

# Download the IMDB dataset
train_data, validation_data, test_data = tfds.load(
    name='imdb_reviews',
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

# sample printing

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch)

print(train_labels_batch)

# Build the model
''' 1) embedding 2) transfer learning -> pre-trained text embedding model available at the TensorFlow Hub '''

# Using the pretrained model, map a sentence to its embedding vector
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"  #(None, 50)
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
print(hub_layer(train_examples_batch[:3]))

lstm_dim = 64 #reshaping the hub_layer is required.

model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Reshape((50, 1)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)
])

model.summary()

# Compile and Train the model
opt = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(train_data.shuffle(10000).batch(256), epochs=30, validation_data=validation_data.batch(256), verbose=1)
plot_loss(history)

# Evaluate the model
results = model.evaluate(test_data.batch(256), verbose=2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" %(name, value))




