# Protocol buffers for efficient serialization of structured data
# messages defined by .proto files
import tensorflow as tf

import numpy as np
import IPython.display as display

#tf.train.Example
# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

# Proto message functions
# Returns a bytes_list from a string / byte
def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# Returns a float_list from a float / double."""
def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# Returns an int64_list from a bool / enum / int / uint.  checking the data type too.
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

print(_bytes_feature(b'test_string'))

print(_bytes_feature(u'test_bytes'.encode('utf-8')))

print(_float_feature(np.exp(1)))

print(_int64_feature(True))
print(_int64_feature(1))

# below two will raise exceptions
try:
    print(_int64_feature(1.0))
except:
    print('Types is wrong.')

try:
    print(_float_feature('Quit!'))
except:
    print('Types is wrong.')

# all protos can be serialized to a binary string using the .SerializeToString method
feature = _float_feature(np.exp(1))
print(type(feature))
print(feature.SerializeToString())

# Creating a tf.train.Example message
# The number of observations in the dataset.
n_observations = int(1e4)

# Boolean feature, encoded as False or True.
feature0 = np.random.choice([False, True], n_observations)

# Integer feature, random from 0 to 4.
feature1 = np.random.randint(0, 5, n_observations)

# String feature.
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]

# Float feature, from a standard normal distribution.
feature3 = np.random.randn(n_observations)

# Creates a tf.train.Example message ready to be written to a file.
def serialize_example(feature0, feature1, feature2, feature3):
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible data type.
  feature = {
      'feature0': _int64_feature(feature0),
      'feature1': _int64_feature(feature1),
      'feature2': _bytes_feature(feature2),
      'feature3': _float_feature(feature3),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


# Creating a message for an observation recored as [False, 4, bytes('goat'), 0.9876] for instance
# This is an example observation from the dataset.

example_observation = []

serialized_example = serialize_example(False, 4, b'goat', 0.9876)
print('Serialized example,', serialized_example)

# Decode the message
example_proto = tf.train.Example.FromString(serialized_example)
print('Example proto, ', example_proto)

'''
# TFRecords format details - each record stored in the following formats:
uint64 length
uint32 masked_crc32_of_length
byte   data[length]
uint32 masked_crc32_of_data

masked_crc = ((crc>>15) | (crc << 17)) + 0xa282ead8ul
'''

# TFRecords files using tf.data
# writing a TFRecord file
print('tf.data.Dataset.from_tensor_slices,', tf.data.Dataset.from_tensor_slices(feature1))
features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
print('features_dataset,', features_dataset)

# Use `take(1)` to only pull one example from the dataset.
for f0,f1,f2,f3 in features_dataset.take(1):
  print(f0)
  print(f1)
  print(f2)
  print(f3)

# Use the tf.data.Dataset.map method to apply a function to each element of a Dataset.
def tf_serialize_example(f0,f1,f2,f3):
  tf_string = tf.py_function(
    serialize_example,
    (f0, f1, f2, f3),  # Pass these args to the above function.
    tf.string)      # The return type is `tf.string`.
  return tf.reshape(tf_string, ()) # The result is a scalar.

print('Serialize the example, ', tf_serialize_example(f0, f1, f2, f3))

# apply the function to each element in the dataset
serialized_features_dataset = features_dataset.map(tf_serialize_example)
print('Map the dataset,', serialized_features_dataset)

def generator():
  for features in features_dataset:
    yield serialize_example(*features)

serialized_features_dataset = tf.data.Dataset.from_generator(
    generator, output_types=tf.string, output_shapes=())

print("Serialized the dataset, ", serialized_features_dataset)

# Write them to a TFRecord file:
filename = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)

# Reading a TFRecord file
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
print(raw_dataset)

for raw_record in raw_dataset.take(10):
  print(repr(raw_record))

# create a description of the features first since tf.data.Dataset uses the graph execution
# Create a description of the features.
feature_description = {
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
}

def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)

parsed_dataset = raw_dataset.map(_parse_function)
print(parsed_dataset)

for parsed_record in parsed_dataset.take(10):
  print(repr(parsed_record))

# TFRecord files in Python

# Write the `tf.train.Example` observations to the file.
with tf.io.TFRecordWriter(filename) as writer:
  for i in range(n_observations):
    example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
    writer.write(example)

# Read a TFRecord file
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
print('Raw dataset, ', raw_dataset)

for raw_record in raw_dataset.take(2):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)


result = {}
# example.features.feature is the dictionary
for key, feature in example.features.feature.items():
  # The values are the Feature objects which contain a `kind` which contains:
  # one of three fields: bytes_list, float_list, int64_list

  kind = feature.WhichOneof('kind')
  result[key] = np.array(getattr(feature, kind).value)

print(result)

# Walkthrough: Reading and writing image data

# Fetch the images
cat_in_snow = tf.keras.utils.get_file(
    '320px-Felis_catus-cat_on_snow.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')

williamsburg_bridge = tf.keras.utils.get_file(
    '194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')

# Display the images
display.display(display.Image(filename=cat_in_snow))
display.display(display.HTML('Image cc-by: <a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg">Von.grzanka</a>'))
display.display(display.Image(filename=williamsburg_bridge))
display.display(display.HTML('<a "href=https://commons.wikimedia.org/wiki/File:New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg">From Wikimedia</a>'))

from PIL import Image
import requests

def show_image(url):
    #Reference: https://stackoverflow.com/questions/7391945/how-do-i-read-image-data-from-a-url-in-python

    response = requests.get(url, stream=True).raw
    image = Image.open(response)
    image.show()

show_image(url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
show_image('https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')

# Write the TFRecord file - encode the raw image string feature and image attributes label feature
image_labels = {
    cat_in_snow : 0,
    williamsburg_bridge : 1,
}

# an example
image_string = open(cat_in_snow, 'rb').read()
label = image_labels[cat_in_snow]

# Create a dictionary with features
def image_example(image_string, label):
    image_shape = tf.io.decode_jpeg(image_string).shape
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

for line in str(image_example(image_string, label)).split('\n')[:15]:
    print(line)
print('...')

# Write the raw image files to 'images.tfrecords'.
# First to 'tf.train.Example' messages and then to a '.tfrecords' file
record_file = 'images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
    for filename, label in image_labels.items():
        image_string = open(filename, 'rb').read()
        tf_example = image_example(image_string, label)
        writer.write(tf_example.SerializeToString())


# Read the TFRecord file
raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

import matplotlib.pyplot as plt

for image_features in parsed_image_dataset:
    print(type(image_features['image_raw'].numpy()))
    image_raw = image_features['image_raw'].numpy()
    image = tf.image.decode_image(image_raw, channels=3)  # default channels = 0.  It messes up the color of the grayscale image.
    plt.imshow(image)
    plt.show()
    # display.display(display.Image(data=image_raw))

plt.close()



