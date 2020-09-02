# path image
import os, os.path
import tensorflow as tf
from PIL import Image

import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


im = Image.open(r"C:\Users\marco\Documents\GitHub\RootPA\App\upload-dir\test.jpg")
im_arr1 = np.array(im) / 255.0

im_arr = np.array([im_arr1])

plt.imshow(im_arr[0])
plt.show()

print(im_arr.shape)

np.save("im_array", im_arr)

