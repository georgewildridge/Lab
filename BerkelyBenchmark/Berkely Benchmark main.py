'''Berkely Benchmark'''
#Input
#Train
import os, sys
from PIL import Image
import PIL as im
import numpy as np
import scipy.io
import theano
import theano.tensor as T
def loadDataset():
	#Train Dataset
	#Loading the Images or X_train values
	train_path_x = '/Users/George/Desktop/BerkelyBenchmark/BSR/BSDS500/data/images/train/'
	train_dirs_x = os.listdir( train_path_x )
	valid_images = [".jpg"]
	train_set_x = []
	rotated = {}
	for file in train_dirs_x:
		ext = os.path.splitext(file)[1]
		file_name = os.path.splitext(file)[0]
		if ext.lower() not in valid_images:
			continue
		img = Image.open(train_path_x + file)
		width, height = img.size
		if width != 481 and height !=321:
			rotated[file_name] = 1
			img = img.rotate(270) #rotates counter clockwise
		else:
			rotated[file_name] = 0 
		train_set_x.append(img)
	#loading the matricies and Y_train values
	train_path_y = '/Users/George/Desktop/BerkelyBenchmark/BSR/BSDS500/data/groundTruth/train/'
	train_dirs_y = os.listdir( train_path_y )
	valid_files = [ ".mat" ]
	train_set_y = []
	for file in train_dirs_y:
		ext = os.path.splitext(file)[1]
		filename = os.path.splitext(file)[0]
		if ext.lower() not in valid_files:
			continue
		train_dataset = np.array(scipy.io.loadmat(train_path_y + filename)["groundTruth"])
		singular_matrix = train_dataset[0][0][0][0][0]
		value = rotated[filename]
		if value == 1:
			singular_matrix = zip(*singular_matrix[::-1])
		train_set_y.append(singular_matrix)
	

	#train_set_x = np.asarray(train_set_x)
	train_set_x = theano.shared(train_set_x)
	train_set_x = T.cast(train_set_x, 'float64')
	train_set_x = T.reshape(train_set_x, [train_set_x.shape[0], train_set_x.shape[1]*train_set_x.shape[2]*train_set_x.shape[3]])
	#train_set_y = np.asarray(train_set_y)
	train_set_y = theano.shared(train_set_y)
	train_set_y = T.cast(train_set_y, 'int32')

	return train_set_y, train_set_x

		
loadDataset()






'''
		train_set_x.append(train_dataset['X'][:, :, :, i])
	train_set_y = train_dataset['y'][:50000,0]
	train_set_x = numpy.asarray(train_set_x)
	train_set_x = theano.shared(train_set_x)
	train_set_x = T.cast(train_set_x, 'float64')
	train_set_x = T.reshape(train_set_x, [train_set_x.shape[0], train_set_x.shape[1]*train_set_x.shape[2]*train_set_x.shape[3]])
	train_set_y = numpy.asarray(train_set_y)
	train_set_y = theano.shared(train_set_y)
	train_set_y = T.cast(train_set_y, 'int32')
	print "Done with creating training dataset"

   




   valid_set_x = []
   for i in range(50001, 73257):
	   valid_set_x.append(train_dataset['X'][:, :, :, i])
   valid_set_y = train_dataset['y'][50001:,0]
   valid_set_x = numpy.asarray(valid_set_x)
   valid_set_x = theano.shared(valid_set_x)
   valid_set_x = T.cast(valid_set_x, 'float64')
   valid_set_x = T.reshape(valid_set_x, [valid_set_x.shape[0], valid_set_x.shape[1]*valid_set_x.shape[2]*valid_set_x.shape[3]])
   valid_set_y = numpy.asarray(valid_set_y)
   valid_set_y = theano.shared(valid_set_y)
   valid_set_y = T.cast(valid_set_y, 'int32')
   print "Done with creating extra dataset"

   test_dataset = scipy.io.loadmat("test_32x32.mat")
   test_set_x = []
   for i in range(0, 26000):
	   test_set_x.append(test_dataset['X'][:,:,:,i])
   test_set_y = test_dataset['y'][:26000,0]
   test_set_x = numpy.asarray(test_set_x)
   test_set_x = theano.shared(test_set_x)
   test_set_x = T.cast(test_set_x, 'float64')
   test_set_x = T.reshape(test_set_x, [test_set_x.shape[0], test_set_x.shape[1]*test_set_x.shape[2]*test_set_x.shape[3]])
   test_set_y = numpy.asarray(test_set_y)
   test_set_y = theano.shared(test_set_y)
   test_set_y = T.cast(test_set_y, 'int32')
   print "Done with creating test dataset"

   print "train_set_x:"
   print train_set_x.eval()
   print "train_set_y:"
   print train_set_y.eval()

   return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y'''
