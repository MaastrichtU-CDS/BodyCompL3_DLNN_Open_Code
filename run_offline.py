import tensorflow as tf
import numpy as np
import os
import datetime
import random
from tensorflow.keras import losses
from shutil import copyfile
import tensorflow_addons as tfa

import models
import utils
import data_augmentation


def run_once(f):
    """
    Wrapper for functions that should only run once every run.

    Parameters
    ----------
    f : function
        Function to be ran.

    Returns
    -------
    wrapper : boolean
    """

    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


class DiceMetric(tf.keras.metrics.Metric):
    def __init__(self, name='dice_coefficient', **kwargs):
        super(DiceMetric, self).__init__(name=name, **kwargs)
        self.dice_score = self.add_weight(name='dsc', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        smooth = 0.000001
        union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        score = tf.reduce_mean((2. * intersection) / (union + smooth), axis=0)
        self.dice_score.assign(score)

    def result(self):
        return self.dice_score

    def reset_states(self):
        self.dice_score.assign(0.0)


def dice_loss(y_true, y_pred):
    smooth = 1
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    score = tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return 1 - score



def early_stopping(loss_list, min_delta=0.005, patience=20):
    """

    Parameters
    ----------
    loss_list : list
        List containing loss values for every evaluation.
    min_delta : float
        Float serving as minimum difference between loss values before early stopping is considered.
    patience : int
        Training will not be stopped before int(patience) number of evaluations have taken place.

    Returns
    -------

    """
    # TODO: Changed to list(loss_list)
    if len(list(loss_list)) // patience < 2:
        return False

    mean_previous = np.mean(loss_list[::-1][patience:2 * patience])
    mean_recent = np.mean(loss_list[::-1][:patience])
    delta_abs = np.abs(mean_recent - mean_previous)  # abs change
    delta_abs = np.abs(delta_abs / mean_previous)  # relative change

    if delta_abs < min_delta:
        print('Stopping early...')
        return True
    else:
        return False


@run_once
def _start_graph_tensorflow():
    """
    Starts the tensorboard graph. Allows for the tracking of loss curves, accuracy and architecture visualization.
    """
    tf.summary.trace_on(graph=True, profiler=True)


@run_once
def _end_graph_tensorflow(self, log_dir):
    """

    Parameters
    ----------
    self : tf.writer
        train_summary_writer.
    log_dir : str
        Path to directory where updates should be stored.

    Returns
    -------

    """
    with self.as_default():
        tf.summary.trace_export(name="graph", step=0, profiler_outdir=log_dir)


def get_batch(images, labels, params):
    ct_batch = np.zeros(shape=[params.dict['batch_size'],
                               params.dict['patch_shape'][0],
                               params.dict['patch_shape'][1],
                               params.dict['patch_shape'][2]])
                               
    gt_batch = np.zeros(shape=[params.dict['batch_size'],
                               params.dict['patch_shape'][0],
                               params.dict['patch_shape'][1],
                               params.dict['patch_shape'][2]])                          
                              
    
    for patch in range(0, params.dict['batch_size']):
        patient = random.randint(0, images.shape[-1] - 1)
        ct = images[:, :, patient]
        gt = labels[:, :, patient]
        # ct = utils.normalize(ct, 'True', params.dict['min_bound'], params.dict['max_bound'])
        #if random.randint(0, 1) == 1:
        #    num_augments = np.random.randint(1, params.dict['number_of_augmentations'] + 1)
        #    ct, gt = data_augmentation.apply_augmentations(ct,
        #                                                   gt,
        #                                                   num_augments)

        ct_batch[patch, :, :, 0] = ct
        gt_batch[patch, :, :, 0] = gt
    gt_batch = tf.one_hot(np.uint8(np.squeeze(gt_batch, axis=-1)), params.dict['num_classes'])
    return ct_batch, gt_batch


def main():
    @tf.function
    def train_on_batch(im_src, gt_src):
        """
        Manages and updates parameters for training.
        Parameters
        ----------
        im_src : np.ndarray
        gt_src : np.ndarray

        Returns
        -------

        """
        with tf.GradientTape() as tape:
            predictions = model(inputs=[im_src], training=True)
            regularization_loss = tf.math.add_n(model.losses)
            loss_value = loss_function(gt_src, predictions)
            total_loss = regularization_loss + loss_value

        grads = tape.gradient(total_loss, model.trainable_weights)
        optimizer_function.apply_gradients(zip(grads, model.trainable_weights))
        train_loss(total_loss)
        return predictions

    @tf.function
    def validate_on_batch(im_src, gt_src):
        """
        Manages validation.

        Parameters
        ----------
        im_src : np.ndarray
        gt_src : np.ndarray

        Returns
        -------

        """
        predictions = model(inputs=[im_src], training=False)
        regularization_loss = tf.math.add_n(model.losses)
        loss_value = loss_function(gt_src, predictions)
        total_loss = regularization_loss + loss_value
        validation_loss(total_loss)
        return predictions

    param_path = os.getcwd() + '/params.json'
    params = utils.Params(param_path)

    # Define loss function
    loss_list = []
    # loss_function = dice_loss
    loss_function = losses.CategoricalCrossentropy()
    # loss_function = tfa.losses.SigmoidFocalCrossEntropy()

    # Define optimizer with learning rate
    optimizer_function = tf.keras.optimizers.Adam(params.dict['learning_rate'])
    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(params.dict['learning_rate'],
    #                                                             decay_steps=params.dict['decay_steps'],
    #                                                             decay_rate=params.dict['decay_rate'],
    #                                                             staircase=True)
    # optimizer_function = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Define model
    #unet
    model = models.unet(params,
                        params.dict['num_classes'],
                        optimizer=optimizer_function,
                        loss=loss_function)

    # Define evaluation metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = DiceMetric()
    validation_loss = tf.keras.metrics.Mean(name='validation_loss')
    validation_accuracy = DiceMetric()

    # Create variables for various paths used for storing training information
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(params.dict['log_path']):
        os.mkdir(params.dict['log_path'])
            
    train_log_dir = params.dict['log_path'] + '/gradient_tape/' + current_time + '/train'
    val_log_dir = params.dict['log_path'] + '/gradient_tape/' + current_time + '/val'
    saved_model_path = params.dict['log_path'] + '/gradient_tape/' + current_time + '/saved_models/'
    saved_weights_path = params.dict['log_path'] + '/gradient_tape/' + current_time + '/saved_weights/'
    
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    os.mkdir(saved_model_path)
    os.mkdir(saved_weights_path)
    # Ralph: this copy statement doesn't work because it assumes that params.dict['log_path'] is a relative path whereas
    # in fact, I set it to an explicit path in the /mnt/localscratch folder. Even as a relative path, it does not seem
    # correct to use os.getcwd() in the destination
    #copyfile(os.getcwd() + '/params.json', 
    #         os.getcwd() + '/' + params.dict['log_path'] + '/gradient_tape/' + current_time + '/params.json')
    copyfile(os.getcwd() + '/params.json', 
             params.dict['log_path'] + '/gradient_tape/' + current_time + '/params.json')

    # Load training and validation data
    images_t, labels_t, names_t = utils.load_dataset(params.dict['data_path_train'], params)
    images_v, labels_v, names_v = utils.load_dataset(params.dict['data_path_val'], params)

    
    # col_idx = np.random.RandomState(seed=42).permutation(images.shape[2])
    # images = images[:, :, col_idx]
    # labels = labels[:, :, col_idx]
    
    # index = int(np.shape(images)[2] * 0.8)
    # images_t = images[:, :, :index]
    # labels_t = labels[:, :, :index]
    # images_v = images[:, :, index:]
    # labels_v = labels[:, :, index:]
    
    print('Training set size: ', np.shape(images_t)[2])
    print('Validation set size: ', np.shape(images_v)[2])

    # Start training loop
    for iteration in range(0, params.dict['num_steps'] + 1):
        _start_graph_tensorflow()
        
        ct_batch, gt_batch = get_batch(images_t, labels_t, params)
        # gt_batch = tf.one_hot(np.uint8(np.squeeze(gt_batch)), params.dict['num_classes'])
        train_pred = train_on_batch(ct_batch, gt_batch)
        
        _end_graph_tensorflow(train_summary_writer, train_log_dir)

        # Evaluation step during training. 
        if iteration % params.dict['train_eval_step'] == 0:
            # Write training information to training log
            with train_summary_writer.as_default():
                train_dice = train_accuracy(gt_batch, train_pred).numpy()
                tf.summary.scalar('loss', train_loss.result(), step=iteration)
                tf.summary.scalar('accuracy', train_dice, step=iteration)
                

            template = 'Iteration {}, Loss: {:.5}, Dice: {:.5}'
            print(template.format(iteration + 1,
                                  train_loss.result(),
                                  train_dice))
                    
        # Evaluation step for validation.
        if iteration % params.dict['val_eval_step'] == 0:
            ct_batch_val, gt_batch_val = get_batch(images_v, labels_v, params)

            # gt_batch_val = tf.one_hot(np.uint8(np.squeeze(gt_batch_val)), params.dict['num_classes'])
            val_pred = validate_on_batch(ct_batch_val, gt_batch_val)

            # Write validation information to log
            with val_summary_writer.as_default():
                validation_dice = validation_accuracy(gt_batch_val, val_pred).numpy()
                tf.summary.scalar('loss', validation_loss.result(), step=iteration)
                tf.summary.scalar('accuracy', validation_dice, step=iteration)
                loss_list.append(validation_loss.result())

            template = 'Iteration {}, Validation Loss: {:.5}, Validation Dice: {:.5}'
            print(template.format(iteration + 1,
                                  validation_loss.result(),
                                  validation_dice))
            
            # Earling stopping when loss in the past 'patience' train_eval_steps
            # is smaller than 'min_delta'. Breaks loop.
            early_stop = early_stopping(loss_list, min_delta=0.001, patience=10)
            if early_stop:
                print("Early stopping signal received at iteration = %d/%d" % (iteration, params.dict['num_steps']))
                print("Terminating training ")
                model.save(os.path.join(saved_model_path,
                                        'model_' + str(iteration)))
                model.save_weights(os.path.join(saved_weights_path,
                                                'model_weights' + str(iteration) + '.h5'))
                break
        
        # Save the model at predefined step numbers.
        if iteration % params.dict['save_model_step'] == 0:
                model.save(os.path.join(saved_model_path,
                                        'model_' + str(iteration)))
                model.save_weights(os.path.join(saved_weights_path,
                                                'model_weights' + str(iteration) + '.h5'))


if __name__ == '__main__':
    # Small check for GPU usage or CPU usage. CUDA_VISIBLE_DEVICES selects a
    # specific GPU card. Usefull when multiple people are training on the
    # same server.
    # CUDA_VISIBLE_DEVICES = 2
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Running on CPU. Please install GPU version of TF")
    main()
