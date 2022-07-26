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

titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
print(titanic.info())
print(titanic.head())

titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')
print(titanic.info())  # each column shows a various data type that is not good for the model fitting.

# Example of a symbolic tensor

input = tf.keras.Input(shape=(), dtype=tf.float32)

# Perform a calculation using the input
result = 2*input + 1

calc = tf.keras.Model(inputs=input, outputs=result)
print(calc(1).numpy())
print(calc(2).numpy())


# Build a preprocessing model

inputs = {}

for name, column in titanic_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
    print(type(inputs[name]))

print(type(inputs), inputs)

# Functional API format should be used since multiple number of inputs are given.
# Normalized numeric inputs and the embedded or one hot encoded categorical inputs will be processed and concatenated within the model

# concatenate the numeric inputs together, and run them through a normalization layer:
#
numeric_inputs = {name: _input for name, _input in inputs.items()
                  if _input.dtype == tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

print(all_numeric_inputs)

# Collect the symbolic preprocessing results and concatenate them later:
preprocessed_inputs = [all_numeric_inputs]

# Use tf.keras.layers.StringLookup to map from strings to integer or tf.keras.layers.Embedding
for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue
    lookup = layers.StringLookup(vocabulary=np.unique(titanic_features[name]))
    one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)

preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

tf.keras.utils.plot_model(model=titanic_preprocessing, rankdir="LR", dpi=72, show_shapes=True)

# convert to the dictionary of tensors
titanic_features_dict = {name: np.array(value)
                         for name, value in titanic_features.items()}

features_dict = {name:values[:1] for name, values in titanic_features_dict.items()}
print(titanic_preprocessing(features_dict))

def titanic_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        layers.Dense(64),
        layers.Dense(1)
    ])

    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam())
    return model

titanic_model = titanic_model(titanic_preprocessing, inputs)

titanic_model.fit(x=titanic_features_dict, y=titanic_labels, epochs=10)

# save the model and reload
titanic_model.save('test')
reloaded = tf.keras.models.load_model('test')

features_dict = {name:values[:1] for name, values in titanic_features_dict.items()}

before = titanic_model(features_dict)
after = reloaded(features_dict)
assert (before-after)<1e-3
print(before)
print(after)

# Using tf.data  (tf.data: Build TensorFlow input pipelines guide, https://www.tensorflow.org/guide/data)

# On in memory data
import itertools

def slices(features):
    for i in itertools.count():
        example = {name:values[i] for name, values in features.items()}
        yield example

for example in slices(titanic_features_dict):
    for name, value in example.items():
        print(f'{name:19s}: {value}')
    break

# Basic in memory dataloader : tf.data.Dataset.from_tensor_slices
features_ds = tf.data.Dataset.from_tensor_slices(titanic_features_dict)

for example in features_ds:
    for name, value in example.items():
        print(f"{name:19s}: {value}")
    break

titanic_ds = tf.data.Dataset.from_tensor_slices((titanic_features_dict, titanic_labels))

# dataset prepared instead of features and labels
titanic_batches = titanic_ds.shuffle(len(titanic_labels)).batch(32)

titanic_model.fit(titanic_batches, epochs=5)

# From a single file
titanic_file_path = tf.keras.utils.get_file('train.csv', 'https://storage.googleapis.com/tf-datasets/titanic/train.csv')

titanic_csv_ds = tf.data.experimental.make_csv_dataset(
    titanic_file_path,
    batch_size=5, # Artificially small to make examples easier to show.
    label_name='survived',
    num_epochs=1,
    ignore_errors=True,)

for batch, label in titanic_csv_ds.take(1):
  for key, value in batch.items():
    print(f"{key:20s}: {value}")
  print()
  print(f"{'label':20s}: {label}")