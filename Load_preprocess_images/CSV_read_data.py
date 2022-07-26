import pandas as pd
import numpy as np

np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow.keras import layers

url = 'https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv'
columns = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
           'Viscera weight', 'shell weight', "Age"]

# Load the data from the CSV file at the url site
abalone_train = pd.read_csv(url, names=columns)
print(abalone_train.info())
print(abalone_train.head())

# Generate feature data and labels
abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')

abalone_features = np.array(abalone_features)
print(abalone_features.shape)

abalone_model = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
])
abalone_model.compile(loss='mean_absolute_error',
                       optimizer=tf.keras.optimizers.Adam())
abalone_model.fit(abalone_features, abalone_labels, epochs=10)

# normalizing
normalize = layers.Normalization()
normalize.adapt(abalone_features)

norm_abalone_model = tf.keras.Sequential([
  normalize,
  layers.Dense(32),
  layers.Dense(32),
  layers.Dense(1)
])

norm_abalone_model.compile(loss='mean_absolute_error',
                           optimizer = tf.keras.optimizers.Adam())

history = norm_abalone_model.fit(abalone_features, abalone_labels, validation_split=0.2,  epochs=20)


import matplotlib.pyplot as plt
def plot_loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()
    #plt.close()

plot_loss(history)