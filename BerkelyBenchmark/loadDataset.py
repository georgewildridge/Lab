import os, sys
from PIL import Image
import PIL as im
import numpy
import scipy.io
import theano
import theano.tensor as T
import timeit
from logistic_sgd import LogisticRegression, load_data
def loadDataset(x_path, y_path):
	#Train Dataset
	#Loading the Images or X_values
	dirs_x = os.listdir( x_path )
	valid_images = [".jpg"]
	set_x = []
	rotated = {}
	for file in dirs_x:
		ext = os.path.splitext(file)[1]
		file_name = os.path.splitext(file)[0]
		if ext.lower() not in valid_images:
			continue
		img = Image.open(x_path + file)
		width, height = img.size
		if width != 481 and height !=321:
			rotated[file_name] = 1
			img = img.rotate(270) #rotates counter clockwise
		else:
			rotated[file_name] = 0 
		set_x.append(img)
	#loading the matricies and Y_values
	dirs_y = os.listdir( y_path )
	valid_files = [ ".mat" ]
	set_y = []
	for file in dirs_y:
		ext = os.path.splitext(file)[1]
		filename = os.path.splitext(file)[0]
		if ext.lower() not in valid_files:
			continue
		dataset = numpy.array(scipy.io.loadmat(y_path + filename)["groundTruth"])
		singular_matrix = dataset[0][0][0][0][0]
		value = rotated[filename]
		if value == 1:
			singular_matrix = zip(*singular_matrix[::-1])
		set_y.append(singular_matrix)
	return set_x, set_y

train_path_x, train_path_y = loadDataset('/Users/George/Desktop/BerkelyBenchmark/BSR/BSDS500/data/images/train/','/Users/George/Desktop/BerkelyBenchmark/BSR/BSDS500/data/groundTruth/train/')
valid_set_x, valid_set_y = loadDataset('/Users/George/Desktop/BerkelyBenchmark/BSR/BSDS500/data/images/val/','/Users/George/Desktop/BerkelyBenchmark/BSR/BSDS500/data/groundTruth/val/')
test_set_x, test_set_y = loadDataset('/Users/George/Desktop/BerkelyBenchmark/BSR/BSDS500/data/images/test/','/Users/George/Desktop/BerkelyBenchmark/BSR/BSDS500/data/groundTruth/test/')

