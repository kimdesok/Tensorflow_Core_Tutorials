import pandas as pd
import numpy as np

np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow.keras import layers

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

# get the data file
traffic_volume_csv_gz = tf.keras.utils.get_file(
    'Metro_Interstate_Traffic_Volume.csv.gz',
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz",
    cache_dir='.', cache_subdir='traffic')

# read directly from the compressed file
traffic_volume_csv_gz_ds = tf.data.experimental.make_csv_dataset(
    traffic_volume_csv_gz,
    batch_size=256,
    label_name='traffic_volume',
    num_epochs=1,
    compression_type='GZIP'
)

for batch, label in traffic_volume_csv_gz_ds.take(1):
    for key, value in batch.items():
        print(f'{key:20s}: {value[:5]}')
    print()
    print(f"{'label':20s}: {label[:5]}")

# Caching - parsing the data once and store them
from time import time

stime = time()
for i, (batch, label) in enumerate(traffic_volume_csv_gz_ds.repeat(20)):
    if i%40 == 0:
        print('.', end='')
print()

print('without caching,', time() - stime)

stime = time()
caching = traffic_volume_csv_gz_ds.cache().shuffle(1000)

for i, (batch, label) in enumerate(caching.shuffle(1000).repeat(20)):
  if i % 40 == 0:
    print('.', end='')
print()

print('with caching,', time() - stime)

stime = time()
snapshot = tf.data.experimental.snapshot('titanic.tfsnap')
snapshotting = traffic_volume_csv_gz_ds.apply(snapshot).shuffle(1000)

for i, (batch, label) in enumerate(snapshotting.shuffle(1000).repeat(20)):
  if i % 40 == 0:
    print('.', end='')
print()

print('with snapshot,', time() - stime)

# EDA
