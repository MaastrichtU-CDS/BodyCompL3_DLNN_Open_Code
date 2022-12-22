import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout, \
    concatenate, BatchNormalization, Conv2DTranspose, PReLU


def conv_block(inputs, num_features, kernel_size, params):
    x = Conv2D(num_features, kernel_size, activation=None, kernel_initializer='he_normal',
               padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=params.l2_loss))(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x = BatchNormalization()(x)
    x = Dropout(params.dropout_rate)(x)
    x = Conv2D(num_features, kernel_size, activation=None, kernel_initializer='he_normal',
               padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=params.l2_loss))(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = BatchNormalization()(x)
    return x


def unet(params, num_classes, optimizer, loss):
    input_ct = Input((None, None, params.dict['patch_shape'][-1]),
                     name='CT_input')
    x_1 = conv_block(input_ct, 32, (3, 3), params)
    p_1 = MaxPooling2D((2, 2))(x_1)

    x_2 = conv_block(p_1, 64, (3, 3), params)
    p_2 = MaxPooling2D((2, 2))(x_2)

    x_3 = conv_block(p_2, 128, (3, 3), params)
    p_3 = MaxPooling2D((2, 2))(x_3)

    x_4 = conv_block(p_3, 256, (3, 3), params)
    p_4 = MaxPooling2D((2, 2))(x_4)

    x_5 = conv_block(p_4, 512, (3, 3), params)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x_5)
    u6 = concatenate([u6, x_4])
    x_6 = conv_block(u6, 256, (3, 3), params)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x_6)
    u7 = concatenate([u7, x_3])
    x_7 = conv_block(u7, 128, (3, 3), params)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x_7)
    u8 = concatenate([u8, x_2])
    x_8 = conv_block(u8, 64, (3, 3), params)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(x_8)
    u9 = concatenate([u9, x_1])
    x_9 = conv_block(u9, 32, (3, 3), params)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(x_9)

    model = Model(inputs=[[input_ct]], outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model
