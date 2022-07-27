import os
import json
import pprint
import tensorflow as tf
import matplotlib.pyplot as plt

# COCO2017 dataset
'''
id:int
image_id: int
category_id: int
segmentation: RLE, object segmentation mask
bbox: [x,y,width, height] object bounding box coordinates
area: float, area of the bounding box
iscrowd: 0 for single 1 for a collection of objects
'''

root_dir = 'datasets'
tfrecords_dir = 'tfrecords'
images_dir = os.path.join(root_dir, 'val2017')
annotations_dir = os.path.join(root_dir, 'annotations')
annotations_file = os.path.join(annotations_dir, 'instances_val2017.json')
images_url = 'http://images.cocodataset.org/zips/val2017.zip'
annotations_url = (
    'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
)

# Download image files
if not os.path.exists(images_dir):
    image_zip = tf.keras.utils.get_file(
        'images.zip', cache_dir=os.path.abspath('.'), origin=images_url, extract=True,
    )
    os.remove(image_zip)

# Download caption annotation files
if not os.path.exists(annotations_dir):
    annotation_zip = tf.keras.utils.get_file(
        'captions.zip',
        cache_dir = os.path.abspath('.'),
        origin=annotations_url,
        extract=True,
    )
    os.remove(annotation_zip)

    print('The COCO dataset has been downloaded and extracted successfully.')


