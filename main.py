# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import os
import matplotlib.pyplot as plt
import k_map_data
from model_architectures.model_man import model_man
import timeit
import argparse

if __name__ == '__main__':

	model_arch = ['mnist']
	dataset_dir = 'dataset/test/'
	model_direc = 'saved_models/'
	check_direc = 'training_checkpoints/'
	check_period = 5
	epochs = [5]
	batches = [5]
	load_model = ''
	resume_check = ''
	model_name = ''
	

	parser = argparse.ArgumentParser()
	parser.add_argument("-a", "--model_arch", nargs='+', type=str, help="Choose the CNN architecture to use {mnist, alex, vgg} [DEFAULT] = mnist")
	parser.add_argument("-d", "--datset_direc", help="Choose the directory for the data [DEFAULT] = dataset/test/")
	parser.add_argument("-m", "--model_direc", help="Choose the directory to save the trained model to [DEFAULT] = saved_models/")
	parser.add_argument("-c", "--check_direc", help="Choose the directory to save the training checkpoints to [DEFAULT] = training_checkpoints/")
	parser.add_argument("-p", "--check_period", help="Choose how often to save a checkpoint [DEFAULT] = 5")
	parser.add_argument("-e", "--num_epoch", nargs='+', type=int, help="Choose the number of epochs for training (Note: If multiple values are given, a model will be trained with each) [DEFAULT] = [5]")
	parser.add_argument("-b", "--batch_size", nargs='+', type=int, help="Choose the batch size for training (Note: If multiple values are given, a model will be trained with each) [DEFAULT] = [5]")
	parser.add_argument("-n", "--model_name", help="Choose the name to save the trained model as [DEFAULT] = e[num_epoch]_s_[epoch_step]_model_latest.h5")
	parser.add_argument("-l", "--load_model", help="Load trained h5 model")
	parser.add_argument("-r", "--resume", help="Resume from specified checkpoint")
	parser.add_argument("-v", "--verbose", action='store_true', help="Flag to write evaluation results to the file results_cnn.txt Note: This file will be overwritten")
	args = parser.parse_args()
	

	if args.model_arch:
		model_arch = args.model_arch
	if args.datset_direc:
		dataset_dir = args.datset_direc
	if args.model_direc:
		model_direc = args.model_direc
	if args.check_direc:
		check_direc = args.check_direc
	if args.check_period:
		check_period = int(args.check_period)
	if args.num_epoch:
		epochs = args.num_epoch
	if args.batch_size:
		batches = args.batch_size
	if args.model_name:
		model_name = args.model_name
	if args.load_model:
		if args.load_model != '':
			load_model = args.load_model
		else:
			print('h5 file must be specified')
			exit()
	if args.resume:
		if args.resume != '':
			resume_check = args.resume
		else:
			print('checkpoint file must be specified')
			exit()

	(train_images, train_labels),(test_images, test_labels) = k_map_data.get_data(dataset_dir, 0.8, 0.2)
	print('train shape = {}'.format(train_images.shape))
	dim = (256, 256, 1)
	num_cat = 2
	checkpoint_path = check_direc + 'cnn_model_cp-{epoch:04d}.ckpt'
	cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
	                                                 save_weights_only=True,
	                                                 verbose=1,
	                                                 period=check_period)

	train_num = train_images.shape[0]
	train_images = train_images.reshape(train_num, dim[0], dim[1], 1)
	test_num = test_images.shape[0]
	test_images = test_images.reshape(test_num, dim[0], dim[1], 1)

	train_images = train_images / 255.0
	test_images = test_images / 255.0

	num_mig = 0
	num_aig = 0
	num_mig_test = 0
	num_aig_test = 0
	index = 0

	# Train
	for label in train_labels:
		index = index + 1
		if int(label) == 1:
			num_mig = num_mig + 1
		else:
			num_aig = num_aig + 1

	# Test
	index = 0
	for label in test_labels:
		index = index + 1
		if int(label) == 1:
			num_mig_test = num_mig_test + 1
			num_mig = num_mig + 1
		else:
			num_aig_test = num_aig_test + 1
			num_aig = num_aig + 1

	print('Total num k-map images: ' + str(num_aig + num_mig))
	print('Num AIG: ' + str(num_aig))
	print('Num MIG: ' + str(num_mig))
	results_file = ''
	if args.verbose:
		results_file = open('alex_vgg_results_cnn.txt','w+')
	for arch in model_arch:
		if args.verbose:
			results_file.write('Architecture {}\n'.format(arch))
		for epoch in epochs:
			for batch in batches:
				print('Number of epochs: {} Batch size = {}'.format(epoch, batch))
				if args.verbose:
					results_file.write('Number of epochs: {} Batch size = {}'.format(epoch, batch))
					results_file.write('\n')

				model_name = arch + '_e_' + str(epoch) + '_b_' + str(batch) + '_model_latest.h5'
				models = model_man(dim, num_cat)
				model = keras.Sequential()

				if args.load_model:
					if args.load_model != '':
						model_name = args.load_model
						model = keras.models.load_model(model_name)

						model.compile(optimizer=keras.optimizers.Adam(), 
				              loss=tf.keras.losses.sparse_categorical_crossentropy,
				              metrics=['accuracy'])
					else:
						print('h5 file must be specified')
						exit()
				else:
					if arch == 'mnist':
						model = models.create_mnist()
					elif arch == 'alex':
						model = models.create_alex()
					elif arch == 'vgg':
						model = models.create_vgg()
					else:
						print('Invalid model architecture [mnist, alex, resnet]')
						exit()
					
					print(model.summary())
					model.fit(train_images, train_labels, epochs=epoch, batch_size=batch)

				train_loss, train_acc = model.evaluate(train_images, train_labels)
				test_loss, test_acc = model.evaluate(test_images, test_labels)

				print("Trained model, training accuracy: {:5.2f}%".format(100*train_acc))
				print("Trained model, testing accuracy: {:5.2f}%".format(100*test_acc))
				if args.verbose:
					results_file.write("Trained model, training accuracy: {:5.2f}%\n".format(100*train_acc))
					results_file.write("Trained model, testing accuracy: {:5.2f}%\n".format(100*test_acc))
				start = timeit.default_timer()
				predictions = model.predict(test_images)
				stop = timeit.default_timer()
				runtime = stop - start
				print('Time to perform predictions for testing set: ' + str(runtime))
				num = 0
				aig_num = 0
				mig_num = 0
				for i, test in enumerate(test_images):
					
					if np.argmax(predictions[i]) != test_labels[i]:
						num = num + 1
						if np.argmax(predictions[i]) == 0:
							mig_num = mig_num + 1
						else:
							aig_num = aig_num + 1
					
				acc = (((num_aig_test + num_mig_test) - num) /  (num_aig_test + num_mig_test)) * 100
				print('Num AIG in testing set = ' + str(num_aig_test))
				print('Num MIG in testing set = ' + str(num_mig_test))
				print('accuracy: {:5.2f}%'.format(acc))
				print(str(num) + ' failed')
				print(str(aig_num) + ' AIG predictions failed')
				print(str(mig_num) + ' MIG predictions failed')

				aig_acc = ((num_aig_test - aig_num) /  num_aig_test) * 100
				mig_acc = ((num_mig_test - mig_num) /  num_mig_test) * 100
				print('AIG accuracy: {:5.2f}%'.format(aig_acc))
				print('MIG accuracy: {:5.2f}%'.format(mig_acc))
				if args.verbose:
					results_file.write('Num AIG in testing set = {}\n'.format(str(num_aig_test)))
					results_file.write('Num MIG in testing set = {}\n'.format(str(num_mig_test)))
					results_file.write('accuracy: {:5.2f}%\n'.format(acc))
					results_file.write('{} failed\n'.format(str(num)))
					results_file.write('{} AIG predictions failed\n'.format(str(aig_num)))
					results_file.write('{} MIG predictions failed\n'.format(str(mig_num)))
					results_file.write('AIG accuracy: {:5.2f}%\n'.format(aig_acc))
					results_file.write('MIG accuracy: {:5.2f}%\n'.format(mig_acc))
				model.save(model_direc + model_name)
				del model
