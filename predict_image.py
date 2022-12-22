import tensorflow as tf
import tensorflow_addons as tfa
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import area_closing
# from tensorflow.kera
import utils

# Specify entire folder, not saved_model.pb
loaded = tf.keras.models.load_model(r'C:\Users\leroy.volmer\PycharmProjects\BodyCompositionSegmentation\logs\gradient_tape\20201113-112123--150-150\saved_models\model_15000')

param_path = os.getcwd() + '/params.json'
params = utils.Params(param_path)

# Load dataset in memory
patient_path = params.dict['data_path']
images, labels, names = utils.load_dataset(patient_path)

images_t = images[:, :, 10:]
labels_t = labels[:, :, 10:]
images_v = images[:, :, :10]
labels_v = labels[:, :, :10]

# Choose a patient
ct = utils.normalize(images_v[:, :, 0], 'True', params.dict['min_bound'], params.dict['max_bound'])
gt = labels_v[:, :, 0]
# gt = area_closing(gt)


# Expand dimensions to fit in model
ct_layer = np.expand_dims(ct, 0)
ct_layer = np.expand_dims(ct_layer, -1)

# Predict image
pred = loaded.predict([ct_layer])

pred_squeeze = np.squeeze(pred)
pred_max = pred_squeeze.argmax(axis=-1)
pred_healthy = pred[0, :, :, 0]
# pred_healthy[pred_healthy < 0.25] = 0
# pred_healthy[pred_healthy > 0.25] = 1

pred_visc_fat = pred[0, :, :, 1]
# pred_visc_fat[pred_visc_fat < 0.25] = 0
# pred_visc_fat[pred_visc_fat > 0.25] = 1

pred_sub_fat = pred[0, :, :, 2]
# pred_sub_fat[pred_sub_fat < 0.25] = 0
# pred_sub_fat[pred_sub_fat > 0.25] = 1

pred_muscle = pred[0, :, :, 3]
# pred_muscle[pred_muscle < 0.25] = 0
# pred_muscle[pred_muscle > 0.25] = 1

pred_img_max = pred[0, :, :, :].argmax(axis=-1)

# plt.imshow(pred_img_max, cmap='viridis')

