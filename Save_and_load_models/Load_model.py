import os
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

# checkpoint callback to save models
checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

def create_model():
    model = tf.keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model

# get an example dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(-1, 28*28)/255.0
test_images = test_images.reshape(-1, 28*28) / 255.0

# untrained model's performance ~10% accuracy
model = create_model()

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
acc=acc*100
print('Untrained model accuracy, %5.2f ' %acc)

# Load the model
model.load_weights(checkpoint_path)
# Reevaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
acc = acc*100
print('Restored model accuracy, %5.2f ' %acc)