import tensorflow as tf
import tensorflow_datasets as tfds
import utils

print(tf.__version__)

import numpy as np
import os
import matplotlib as mpl
#import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

np.random.seed(42)

os.makedirs(utils.IMAGES_PATH, exist_ok=True)

# Data loading using Keras

# Mnist
mnist = tf.keras.datasets.mnist
(train_image, train_labels), (test_images, test_labels) = mnist.load_data()
utils.show_fig(train_image[0])

# Fashion Mnist
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_image, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
utils.show_fig(train_image[0])

#cifar10
cifar10 = tf.keras.datasets.cifar10
(train_image, train_labels), (test_images, test_labels) = cifar10.load_data()
utils.show_fig(train_image[0])


# Boston housing price for regression - time series
boston_housing = tf.keras.datasets.boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
utils.show_data(x_train[0])

# IMDB sentiment analysis dataset
imdb = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data()

# Retrieve the word index file mapping words to indices
word_index = imdb.get_word_index()
# Reverse the word index to obtain a dict mapping indices to words
inverted_word_index = dict((i, word) for (word, i) in word_index.items())
# Decode the first sequence in the dataset
decoded_sequence = " ".join(inverted_word_index[i] for i in x_train[0])
print(f'\n{decoded_sequence}')

# Reuters newswire classification dataset
reuters = tf.keras.datasets.reuters
(x_train, y_train), (x_test, y_test) = reuters.load_data()

# Retrieve the word index file mapping words to indices
word_index = reuters.get_word_index()
# Reverse the word index to obtain a dict mapping indices to words
inverted_word_index = dict((i, word) for (word, i) in word_index.items())
# Decode the first sequence in the dataset
decoded_sequence = " ".join(inverted_word_index[i] for i in x_train[0])
print(f'\n{decoded_sequence}')


''' 
# Construct a tf.data.Dataset
ds = tfds.load('fashion_mnist', split='train', shuffle_files=True)

# Build your input pipeline
ds = ds.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
for example in ds.take(1):
  image, label = example["image"], example["label"]
  print(np.array(label[0]))
  plt.imshow(image[0])
  plt.show()
'''






