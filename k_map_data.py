import os
import collections
import re
import hashlib
import math
import numpy as np
import tensorflow as tf

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1


def create_image_lists(image_dir, training_percentage, testing_percentage):
		
	if not tf.gfile.Exists(image_dir):
		print("Image directory '" + image_dir + "' not found.")
		return None
	result = collections.OrderedDict()
	sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
	# The root directory comes first, so skip it.
	is_root_dir = True
	for sub_dir in sub_dirs:
		if is_root_dir:
			is_root_dir = False
			continue

		file_list = []
		dir_name = os.path.basename(sub_dir)
		if dir_name == image_dir:
			continue
			
		file_glob = os.path.join(image_dir, dir_name, '*.txt')
		file_list.extend(tf.gfile.Glob(file_glob))
		if not file_list:
			print('No files found')
			continue
		if len(file_list) < 20:
			print('WARNING: Folder has less than 20 images, which may cause issues.')
		elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
			print(
					'WARNING: Folder {} has more than {} images. Some images will '
					'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
		label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
		training_images = []
		testing_images = []
		index = 0
		for file_name in file_list:
			index = index + 1
			base_name = os.path.basename(file_name)

			if index <= (len(file_list) * testing_percentage):
				testing_images.append(base_name)
			else:
				training_images.append(base_name)

		result[label_name] = {
				'dir': dir_name,
				'training': training_images,
				'testing': testing_images,
		}
	return result

def get_data(image_dir, train_percentage, testing_percentage):
	image_lists = create_image_lists(image_dir, train_percentage, testing_percentage)
	class_count = len(image_lists.keys())
	train = []
	_train_labels = []
	test = []
	_test_labels = []
	sess = tf.InteractiveSession()
	for key in image_lists.keys():
		for image in image_lists[key].get('training'):

			key_dir = key.upper()
			key_dir = key_dir.replace(" ", "_")
			matrix = np.fromfile(image_dir + key_dir + '/' + image, dtype='uint8')

			if matrix.shape[0] == 65536:
				_train_labels.append(key)
				matrix = matrix.reshape(256, 256)
				train.append(matrix) 
			
		for image in image_lists[key].get('testing'):
			key_dir = key.upper()
			key_dir = key_dir.replace(" ", "_")
			matrix = np.fromfile(image_dir + key_dir + '/' + image, dtype='uint8')
			
			if matrix.shape[0] == 65536:
				_test_labels.append(key)
				matrix = matrix.reshape(256, 256)
				test.append(matrix) 
			
	sess.close()
	index_train = []
	for key in _train_labels:
		key_dir = key.upper()
		key_dir = key_dir.replace(" ", "_")
		if key_dir == 'ABC_AIG':
			index_train.append(0)
		else:
			index_train.append(1)

	_train_labels = np.vstack(index_train)
	_train_labels = _train_labels.reshape(len(index_train))

	index_test = []
	for key in _test_labels:
		key_dir = key.upper()
		key_dir = key_dir.replace(" ", "_")
		if key_dir == 'ABC_AIG':
			index_test.append(0)
		else:
			index_test.append(1)

	_test_labels = np.vstack(index_test)
	_test_labels = _test_labels.reshape(len(index_test))

	_train_images = np.vstack(train)
	_train_images = _train_images.reshape(len(train), 256, 256)
	_test_images = np.vstack(test)
	_test_images = _test_images.reshape(len(test), 256, 256)

	return (_train_images, _train_labels), (_test_images, _test_labels)




