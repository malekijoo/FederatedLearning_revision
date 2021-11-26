import tensorflow as tf

from config import *


class create_keras_model(tf.keras.Model):

    def __init__(self, name):

        if name=='attention':
            pass
        elif name=='VGG':
            return self.build_vgg()
        elif name=='resnet' or name=='RESNET' or name=='ResNet':
            return tf.keras.applications.ResNet101(include_top=True, weights=None
                                                   )

    def build_vgg(self, input_shape):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(input_shape=(28, 28, 1), filters=64, kernel_size=(3, 3), padding="same",
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
