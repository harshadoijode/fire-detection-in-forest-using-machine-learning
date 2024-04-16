import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
# load the model
model = tf.keras.models.load_model('keras_Model.h5')

with open('labels.txt', 'r') as f:
    label_names = f.read().splitlines()

img_path='12.png'

img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x/255.0



predictions = model.predict(x)
class_index = np.argmax(predictions[0])
class_name_predicted = label_names[class_index]




print('predicted class: ', class_name_predicted)
