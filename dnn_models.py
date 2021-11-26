import tensorflow as tf

from config import *


class keras_model(tf.keras.Model):

    def __init__(self, name):

        if name == 'attention':
            pass

        elif name == 'VGG':
            self.model = self.build_vgg()

        elif name == 'resnet' or name == 'RESNET' or name == 'ResNet':
            print('ResNet model is built with tf.keras.application.ResNet101')
            self.model = tf.keras.applications.ResNet101(include_top=True, weights=None,
                                                         input_shape=input_shape[1:], classes=NumClass,
                                                         classifier_activation='softmax')

    def __call__(self):
        print('loading the model')
        return self.model

    def build_vgg(self):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(input_shape=input_shape[1:], filters=64, kernel_size=(3, 3), padding="same",
                                         activation="relu"))
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=4096, activation="relu"))
        model.add(tf.keras.layers.Dense(units=4096, activation="relu"))
        model.add(tf.keras.layers.Dense(units=NumClass, activation="softmax"))

        ##############################################################################################
        ##############################################################################################

        return model
