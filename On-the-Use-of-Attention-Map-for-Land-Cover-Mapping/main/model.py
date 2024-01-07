# Author: Suesarn Wilainuch

import tensorflow as tf


def define_attention_model(image_shape, num_classes, learning_rate):
    init = tf.keras.initializers.HeUniform()
    in_image = tf.keras.Input(shape=image_shape)
    n_filters = 64

    shortcut1 = tf.keras.layers.Conv2D(n_filters * 2, kernel_size=(1, 1), padding="same", kernel_initializer=init)(
        in_image)
    shortcut1 = tf.keras.layers.BatchNormalization()(shortcut1)
    shortcut1 = tf.keras.layers.Activation('relu')(shortcut1)
    # conv1
    seq = tf.keras.layers.Conv2D(n_filters, kernel_size=(3, 3), padding="same", kernel_initializer=init)(in_image)
    seq = tf.keras.layers.BatchNormalization()(seq)
    seq = tf.keras.layers.Activation('relu')(seq)
    seq = tf.keras.layers.Dropout(0.1)(seq)
    seq = tf.keras.layers.Conv2D(n_filters, kernel_size=(3, 3), padding="same", kernel_initializer=init)(seq)
    seq = tf.keras.layers.BatchNormalization()(seq)
    seq = tf.keras.layers.Activation('relu')(seq)
    # down1
    seq = tf.keras.layers.Conv2D(n_filters, kernel_size=(3, 3), strides=(2, 2), padding="same",
                                 kernel_initializer=init)(seq)
    seq = tf.keras.layers.BatchNormalization()(seq)
    seq = tf.keras.layers.Activation('relu')(seq)
    shortcut2 = tf.keras.layers.Conv2D(n_filters * 4, kernel_size=(1, 1), padding="same", kernel_initializer=init)(seq)
    shortcut2 = tf.keras.layers.BatchNormalization()(shortcut2)
    shortcut2 = tf.keras.layers.Activation('relu')(shortcut2)
    # conv2
    seq = tf.keras.layers.Conv2D(n_filters * 2, kernel_size=(3, 3), padding="same", kernel_initializer=init)(seq)
    seq = tf.keras.layers.BatchNormalization()(seq)
    seq = tf.keras.layers.Activation('relu')(seq)
    seq = tf.keras.layers.Dropout(0.1)(seq)
    seq = tf.keras.layers.Conv2D(n_filters * 2, kernel_size=(3, 3), padding="same", kernel_initializer=init)(seq)
    seq = tf.keras.layers.BatchNormalization()(seq)
    seq = tf.keras.layers.Activation('relu')(seq)
    # down2
    seq = tf.keras.layers.Conv2D(n_filters * 2, kernel_size=(3, 3), strides=(2, 2), padding="same",
                                 kernel_initializer=init)(seq)
    seq = tf.keras.layers.BatchNormalization()(seq)
    seq = tf.keras.layers.Activation('relu')(seq)
    # conv3
    seq = tf.keras.layers.Conv2D(n_filters * 4, kernel_size=(3, 3), padding="same", kernel_initializer=init)(seq)
    seq = tf.keras.layers.BatchNormalization()(seq)
    seq = tf.keras.layers.Activation('relu')(seq)
    seq = tf.keras.layers.Dropout(0.1)(seq)
    seq = tf.keras.layers.Conv2D(n_filters * 4, kernel_size=(3, 3), padding="same", kernel_initializer=init)(seq)
    seq = tf.keras.layers.BatchNormalization()(seq)
    seq = tf.keras.layers.Activation('relu')(seq)

    # up1, merge1
    seq = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(seq)
    seq = tf.keras.layers.BatchNormalization()(seq)
    seq = tf.keras.layers.Activation('relu')(seq)
    seq = tf.keras.layers.add([seq, shortcut2])
    # convT1
    seq = tf.keras.layers.Conv2D(n_filters * 4, kernel_size=(3, 3), padding="same", kernel_initializer=init)(seq)
    seq = tf.keras.layers.BatchNormalization()(seq)
    seq = tf.keras.layers.Activation('relu')(seq)
    seq = tf.keras.layers.Dropout(0.1)(seq)
    seq = tf.keras.layers.Conv2D(n_filters * 2, kernel_size=(3, 3), padding="same", kernel_initializer=init)(seq)
    seq = tf.keras.layers.BatchNormalization()(seq)
    seq = tf.keras.layers.Activation('relu')(seq)
    # up2, merge2
    seq = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(seq)
    seq = tf.keras.layers.BatchNormalization()(seq)
    seq = tf.keras.layers.Activation('relu')(seq)
    seq = tf.keras.layers.add([seq, shortcut1])
    # convT2
    seq = tf.keras.layers.Conv2D(n_filters * 2, kernel_size=(3, 3), padding="same", kernel_initializer=init)(seq)
    seq = tf.keras.layers.BatchNormalization()(seq)
    seq = tf.keras.layers.Activation('relu')(seq)
    seq = tf.keras.layers.Dropout(0.1)(seq)
    seq = tf.keras.layers.Conv2D(n_filters * 2, kernel_size=(3, 3), padding="same", kernel_initializer=init)(seq)
    seq = tf.keras.layers.BatchNormalization()(seq)
    seq = tf.keras.layers.Activation('relu')(seq)

    # feature extraction
    fe = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", kernel_initializer=init)(seq)
    fe = tf.keras.layers.BatchNormalization()(fe)
    fe = tf.keras.layers.Activation('relu', name='feature_map')(fe)

    # Attention layer
    A = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1), strides=(1, 1), padding='same',
                               kernel_initializer=init)(seq)
    A = tf.keras.layers.BatchNormalization()(A)
    A = tf.keras.layers.Activation("relu")(A)
    A = tf.keras.layers.Activation('softmax', name='Attention_layer')(A)

    multi = tf.keras.layers.multiply([fe, A])
    multi = tf.keras.layers.Activation('softmax')(multi)
    GMP = tf.keras.layers.GlobalMaxPooling2D(name='Max')(multi)
    GAP = tf.keras.layers.GlobalAveragePooling2D(name='Avg')(multi)

    output = tf.keras.layers.average([GMP, GAP])

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.Model(in_image, output)
    model.compile(loss=['binary_crossentropy'], optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model
