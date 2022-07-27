import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
pd.set_option('display.max_columns', None)

# The auto MPG dataset from UCI ML repository
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()
print(dataset.tail())
print(dataset.isna().sum())
dataset = dataset.dropna()

# origin is supposed to be categorical
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
print(dataset.head())

# Split the data into train and test sets
print(dataset.shape)
train_dataset = dataset.sample(frac=0.8, random_state=0)
print(train_dataset.shape)
test_dataset = dataset.drop(train_dataset.index)
print(test_dataset.shape)

# Inspect the data

print(train_dataset.describe().transpose())

#sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')  # kernel density estimation at the diagonal

#plt.show()

# Split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

print(train_dataset.describe().transpose()[['mean', 'std']])

# Normalization layer
normalizer = tf.keras.layers.Normalization(axis=-1)

normalizer.adapt(np.array(train_features))

# Normalizer that shows the mean of the train_features
means = {}
for idx, column in enumerate(train_features.columns):
    means[column] = normalizer.mean.numpy()[0][idx]
print(means)

# Normalizer in action
first = np.array(train_features[:1])
with np.printoptions(precision=2, suppress=True):
    print('First example:', first)
    print()
    print("normalized:", normalizer(first).numpy())

# Linear regression

# a linear transformation y = mx + b while x is the horsepower and y is the MPG.
horsepower = np.array(train_features['Horsepower'])
horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)

# build he keras sequential model

horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

horsepower_model.summary()

print(horsepower_model.predict(horsepower[:10]))

# Compile and train the model
horsepower_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    #plt.show()
    #plt.close()

plot_loss(history)

# Collect the results

test_results = {}
test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0
)

x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

def plot_horsepower(x, y):
    plt.figure()
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()
    #plt.show()
    #plt.close()

plot_horsepower(x, y)

# Linear regression with multiple inputs
# y = mx + b where x is a matrix and b is a vector.

linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.predict(train_features[:10])
print(linear_model.layers[1].kernel)

linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error'
)

history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    verbose=0,
    validation_split=0.2
)

plot_loss(history)

test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0
)

# Regression with a deep neural network (DNN)

def build_and_compile_model(norm):
    model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
    return model

# A model with one input data
dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)

dnn_horsepower_model.summary()

history = dnn_horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

plot_loss(history)

# Plotting the prediction based on a nonlinear model

x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)

plot_horsepower(x, y)

test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
    test_features['Horsepower'], test_labels,
    verbose=0)

# Regression using a DNN and multiple inputs

dnn_model = build_and_compile_model(normalizer)

dnn_model.summary()

import time

st=time.time()
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100
)
et = time.time() - st
print(
    'Execution time: %.3f seconds'%et)
plot_loss(history)

# The results on the test set

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

print(pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T)

# Make predictions
test_predictions = dnn_model.predict(test_features).flatten()

plt.figure()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True MPG')
plt.ylabel('Predicted MPG')
lims = [0,50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


error = test_predictions - test_labels
plt.figure()
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')
plt.show()
plt.close()

# Save and reload the model
model_name = 'dnn_model'
dnn_model.save(model_name)

reloaded = tf.keras.models.load_model(model_name)
test_results['reloaded'] = reloaded.evaluate(
    test_features, test_labels, verbose=0)

print(pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T)


