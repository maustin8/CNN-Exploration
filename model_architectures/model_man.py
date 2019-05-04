# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import rmsprop

# Helper libraries
import numpy as np
import os
import matplotlib.pyplot as plt
import timeit

class model_man():
	def __init__(self, dim, num_cat):

		self.dim, self.num_cat = dim, num_cat

	def create_mnist(self):

		model = keras.Sequential([
			#layer 1
	    keras.layers.Flatten(input_shape=self.dim),
	    keras.layers.Dense(250, activation=tf.nn.relu),
	    keras.layers.Dropout(0.5),
	    #layer 2
	    keras.layers.Dense(self.num_cat, activation=tf.nn.softmax)
		])

		model.compile(optimizer=keras.optimizers.Adam(), 
	              loss=tf.keras.losses.sparse_categorical_crossentropy,
	              metrics=['accuracy'])

		return model

	def create_alex(self):

		model = keras.Sequential([
			#layer 1
			keras.layers.Conv2D(96, (11, 11), input_shape=self.dim, padding='same', strides=(4,4)),
			keras.layers.Activation('relu'),
			keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)),
			#layer 2
			keras.layers.Conv2D(256, (3, 3), padding='same', strides=(1,1)),
			keras.layers.Activation('relu'),
			keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)),
			#layer 3
			keras.layers.Conv2D(384, (3, 3), padding='same', strides=(1,1)),
			keras.layers.Activation('relu'),
			#layer 4
			keras.layers.Conv2D(384, (3, 3), padding='same', strides=(1,1)),
			keras.layers.Activation('relu'),
			#layer 5
			keras.layers.Conv2D(256, (3, 3), padding='same', strides=(1,1)),
			keras.layers.Activation('relu'),
			keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)),
			#layer 6
			keras.layers.Flatten(),
			keras.layers.Dense(9216),
			keras.layers.Activation('relu'),
			keras.layers.Dropout(0.5),
			#layer 7
			keras.layers.Dense(4096),
			keras.layers.Activation('relu'),
			keras.layers.Dropout(0.5),
			#layer 8
			keras.layers.Dense(4096),
			keras.layers.Activation('relu'),
			keras.layers.Dropout(0.5),
			#output
			keras.layers.Dense(self.num_cat),
			keras.layers.Activation('softmax')
		])

		model.compile(optimizer=keras.optimizers.Adam(), 
	              loss=tf.keras.losses.sparse_categorical_crossentropy,
	              metrics=['accuracy'])
		
		return model

	def create_vgg(self):

		model = keras.Sequential([
			#layer 1 & 2
			keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=self.dim),
			keras.layers.Activation('relu'),
			keras.layers.ZeroPadding2D((1, 1)),
			keras.layers.Conv2D(64, (3, 3), padding='same'),
			keras.layers.Activation('relu'),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),
			#layer 3 & 4
			keras.layers.ZeroPadding2D((1, 1)),
			keras.layers.Conv2D(128, (3, 3), padding='same'),
			keras.layers.Activation('relu'),
			keras.layers.ZeroPadding2D((1, 1)),
			keras.layers.Conv2D(128, (3, 3), padding='same'),
			keras.layers.Activation('relu'),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),
			#layer 5, 6, & 7
			keras.layers.ZeroPadding2D((1, 1)),
			keras.layers.Conv2D(256, (3, 3), padding='same'),
			keras.layers.Activation('relu'),
			keras.layers.ZeroPadding2D((1, 1)),
			keras.layers.Conv2D(256, (3, 3), padding='same'),
			keras.layers.Activation('relu'),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),
			keras.layers.ZeroPadding2D((1, 1)),
			keras.layers.Conv2D(256, (3, 3), padding='same'),
			keras.layers.Activation('relu'),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),
			#layer 8, 9, & 10
			keras.layers.ZeroPadding2D((1, 1)),
			keras.layers.Conv2D(512, (3, 3), padding='same'),
			keras.layers.Activation('relu'),
			keras.layers.ZeroPadding2D((1, 1)),
			keras.layers.Conv2D(512, (3, 3), padding='same'),
			keras.layers.Activation('relu'),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),
			keras.layers.ZeroPadding2D((1, 1)),
			keras.layers.Conv2D(512, (3, 3), padding='same'),
			keras.layers.Activation('relu'),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),
			#layer 11, 12, & 13
			keras.layers.ZeroPadding2D((1, 1)),
			keras.layers.Conv2D(512, (3, 3), padding='same'),
			keras.layers.Activation('relu'),
			keras.layers.ZeroPadding2D((1, 1)),
			keras.layers.Conv2D(512, (3, 3), padding='same'),
			keras.layers.Activation('relu'),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),
			keras.layers.ZeroPadding2D((1, 1)),
			keras.layers.Conv2D(512, (3, 3), padding='same'),
			keras.layers.Activation('relu'),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),
			#layer 14, 15, & 16
			keras.layers.Flatten(),
			keras.layers.Dense(4096),
			keras.layers.Activation('relu'),
			keras.layers.Dropout(0.5),
			keras.layers.Dense(4096),
			keras.layers.Activation('relu'),
			keras.layers.Dropout(0.5),
			keras.layers.Dense(self.num_cat),
			keras.layers.Activation('softmax')
			
		])

		model.compile(optimizer=keras.optimizers.Adam(), 
	              loss=tf.keras.losses.sparse_categorical_crossentropy,
	              metrics=['accuracy'])
		
		return model