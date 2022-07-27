import os
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

# get an example dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images[:1000].reshape(-1, 28*28)/255.0
test_images = test_images[:1000].reshape(-1, 28*28) / 255.0
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
print(type(train_images), train_images.shape)

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

# checkpoint callback to save models
checkpoint_path = 'training_2/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                save_freq=5*batch_size,
                                                verbose=1)

model=create_model()

model.summary()

model.save_weights(checkpoint_path.format(epoch=0))

model.fit(train_images,
          train_labels,
          epochs=50,
          batch_size=batch_size,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])

print(os.listdir(checkpoint_dir))
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)


# Evaluate with the model saved lastly
# Create a new model instance
model = create_model()

# Load the previously saved weights
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# Manually save weights
model.save_weights('./checkpoints/my_checkpoint')
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')
loss, acc=model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# Save the entire model - useful since it does not require the python code for the model. Two kinds: SavedModel and HDF5
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
model_dir = 'saved_model'
# Check whether the specified path exists or not
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, 'my_model')
model.save(model_path)

print(os.listdir(model_path))

# Reload the model
new_model = tf.keras.models.load_model(model_path)
new_model.summary()

# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))


# Save an entire model as HDF5 format
model = create_model()
model.fit(train_images, train_labels, epochs=5)

model_dir = 'hdf5_model'
# Check whether the specified path exists or not
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, 'my_model.h5')
model.save(model_path)

# Reload the model
new_model = tf.keras.models.load_model(model_path)
new_model.summary()

# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

