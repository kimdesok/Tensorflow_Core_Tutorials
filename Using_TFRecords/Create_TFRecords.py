import os
import json
import pprint
import tensorflow as tf
import matplotlib.pyplot as plt

root_dir = "datasets"
tfrecords_dir = "tfrecords"
images_dir = os.path.join(root_dir, "val2017")
annotations_dir = os.path.join(root_dir, "annotations")
annotation_file = os.path.join(annotations_dir, "instances_val2017.json")
images_url = "http://images.cocodataset.org/zips/val2017.zip"
annotations_url = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)


with open(annotation_file, 'r') as f:
    annotations = json.load(f)['annotations']

print(f'Number of images: {len(annotations)}')

pprint.pprint(annotations[60])

# Parameters

num_samples = 4096
num_tfrecords = len(annotations) // num_samples
if len(annotations) % num_samples:
    num_tfrecords += 1
if not os.path.exists(tfrecords_dir):
    os.makedirs(tfrecords_dir)

# Define TFRecords helpers


def image_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(image, path, example):
    feature = {
        'image': image_feature(image),
        'path': bytes_feature(path),
        'area': float_feature(example['area']),
        'bbox': float_feature_list(example['bbox']),
        'category_id': int64_feature(example['category_id']),
        'id': int64_feature(example['id']),
        'image_id': int64_feature(example['image_id']),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'path': tf.io.FixedLenFeature([], tf.string),
        "area": tf.io.FixedLenFeature([], tf.float32),
        "bbox": tf.io.VarLenFeature(tf.float32),
        "category_id": tf.io.FixedLenFeature([], tf.int64),
        "id": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    example["bbox"] = tf.sparse.to_dense(example["bbox"])
    return example

# Generate data in the TFRecord format
for tfrec_num in range(num_tfrecords):
    samples = annotations[(tfrec_num * num_samples) : ((tfrec_num + 1) * num_samples)]
    with tf.io.TFRecordWriter(
        tfrecords_dir + '/file_%.2i-%i.tfrec'% (tfrec_num, len(samples))
    ) as writer:
        for sample in samples:
            image_path = f"{images_dir}/{sample['image_id']:012d}.jpg"
            image = tf.io.decode_jpeg(tf.io.read_file(image_path))
            example = create_example(image, image_path, sample)
            writer.write(example.SerializeToString())

# Explore a sample from TFRecords
raw_dataset = tf.data.TFRecordDataset(f'{tfrecords_dir}/file_00-{num_samples}.tfrec')
parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

for features in parsed_dataset.take(1):
    for key in features.keys():
        if key != 'image':
            print(f'{key}: {features[key]}')
    print(f'Image shape: {features["image"].shape}')
    plt.figure(figsize=(7, 7))
    plt.imshow(features['image'].numpy())
    plt.show()