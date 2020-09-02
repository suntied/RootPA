# path image
import os, os.path
import tensorflow as tf
from PIL import Image

import numpy as np

path = r"D:\Licence\ALgo_Machie\Projet\2020_3A_IABD2_Correction_Modele_Lineaire_Python\Dataset\DatasetAction"
path2 = r"D:\Licence\ALgo_Machie\Projet\2020_3A_IABD2_Correction_Modele_Lineaire_Python\Dataset\DatasetComedy"
extension = [".png"]
images_action = []

images_actions_class = []

for element in os.listdir(path):
    ext = os.path.splitext(element)[1]
    if ext.lower() not in extension:
        continue
    #image = Image.open(os.path.join(path,element))
    #array = np.array(image) / 255

    image = tf.keras.preprocessing.image.load_img(os.path.join(path,element),
                                                  grayscale=False,
                                                  target_size=(32,32),
                                                  interpolation='nearest')
    images_action.append(tf.keras.preprocessing.image.img_to_array(image, data_format=None, dtype=None) / 255)

    #images_action.append(array.flatten())
    #ToDo add another class to get 3 values 1,-1,-1
    images_actions_class.append(np.array([1,-1,-1]))


print(images_action)

images_comedy = []

images_comedy_class = []

for element in os.listdir(path2):
    ext = os.path.splitext(element)[1]
    if ext.lower() not in extension:
        continue
    # image = Image.open(os.path.join(path_file_img_movie_action,element))
    # array = np.array(image) / 255

    image = tf.keras.preprocessing.image.load_img(os.path.join(path2, element),
                                                  grayscale=False,
                                                  target_size=(32, 32),
                                                  interpolation='nearest')
    images_comedy.append(tf.keras.preprocessing.image.img_to_array(image, data_format=None, dtype=None) / 255)

    # images_action.append(array.flatten())
    # ToDo add another class to get 3 values 1,-1,-1
    images_comedy_class.append(np.array([1, -1,-1]))

print(images_comedy)