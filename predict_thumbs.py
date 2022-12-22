import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import utils


# Load model:
loaded = tf.keras.models.load_model(r'C:\Users\leroy.volmer\PycharmProjects\BCS\logs\gradient_tape\date-time\saved_models\model_28900')
# Load params:
param_path = os.getcwd() + '/params.json'
params = utils.Params(param_path)
# Load images:
images, labels, _ = utils.load_dataset(params.dict['data_path'], params)

# Which image to start and end on, delta should be the same as number of sub-
# plots below
fig = plt.figure()
start_im = 75
end_im = start_im + 25
for idx in range(end_im - start_im):
    # Creates a 5 x 5 subplots
    ax = fig.add_subplot(5, 5, idx + 1)
    
    # Select image
    img = images[:, :, start_im + idx]
    
    # Expand image dimensions for prediction
    img2 = np.expand_dims(img, 0)
    img2 = np.expand_dims(img2, -1)
    
    # Predict
    pred = loaded.predict([img2])
    
    # Remove useless dimensions
    pred_squeeze = np.squeeze(pred)
    
    # Maximum class
    pred_max = pred_squeeze.argmax(axis=-1)
    gt = labels[:, :, start_im + idx]
    
    # Plot image in subplot
    ax.imshow(img, cmap='gray')
    ax.imshow(pred_max, cmap='viridis', alpha=0.3)
    ax.axis('off')

