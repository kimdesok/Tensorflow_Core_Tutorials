import tensorflow as tf
import tensorflow_datasets as tfds

print(tf.__version__)

import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ann"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

mpl.rc('axes', labelsize=12)
mpl.rc('xtick', labelsize=10)
mpl.rc('ytick', labelsize=10)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def show_fig(fig_id):
    plt.figure()
    plt.imshow(fig_id)
    plt.colorbar()
    plt.show()
    plt.close()

def show_data(data_id):
    plt.plot(data_id)
    plt.show()

def check_images(images, class_names, labels, no_images=25):
    plt.figure(figsize=(10, 10))
    nrow = 5
    ncol = int(no_images/nrow + 0.5)
    for i in range(no_images):
        plt.subplot(nrow, ncol, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
        plt.subplots_adjust(hspace=0.4)
    plt.show()
    plt.close()


def plot_image(image, prediction_array, predicted_label, true_label):
    plt.imshow(image, cmap=plt.cm.binary)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                         100 * np.max(prediction_array),
                                         true_label), color=color, fontsize=8)


def plot_value_array(prediction_array, predicted_index, true_index):
    thisplot = plt.bar(range(10), prediction_array, color='#777777')
    plt.ylim([0, 1])

    thisplot[predicted_index].set_color('red')
    thisplot[true_index].set_color('blue')

def check_labels(images, prediction_arrays, class_names, true_labels):
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols

    plt.figure(figsize=(3 * 2 * num_cols, 2 * num_rows))
    plt.rc('xtick', labelsize=8)
    plt.subplots_adjust(hspace=0.8)

    for i in range(num_images):
        image = images[i]
        true_label = class_names[true_labels[i]]
        predicted_label = class_names[np.argmax(prediction_arrays[i])]

        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plt.xticks([])
        plt.yticks([])
        plot_image(image, prediction_arrays[i], predicted_label, true_label)

        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plt.xticks(range(10))
        plt.yticks([])
        plot_value_array(prediction_arrays[i], np.argmax(prediction_arrays[i]), true_labels[i])
        _ = plt.xticks(range(10), class_names, rotation=75)

    save_fig("fmnist_images_labels")
    plt.show()
    plt.close()

def plot_loss(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    save_fig("keras_learning_curves_plot")
    plt.show()
    plt.close()

