#dkj
'''Berkely Benchmark'''
#Input
#Train
import os, sys
from PIL import Image
import PIL as im
import numpy
import scipy.io
import theano
import theano.tensor as T
import timeit
from logistic_sgd import LogisticRegression, load_data
import glob
import cPickle

def loadDataset(x_path, y_path):
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
		img = Image.open(x_path + file).getdata()
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
	i = 0
	for file in dirs_y:
		ext = os.path.splitext(file)[1]
		filename = os.path.splitext(file)[0]
		if ext.lower() not in valid_files:
			continue
		dataset = numpy.array(scipy.io.loadmat(y_path + filename)["groundTruth"])
		singular_matrix = dataset[0][0][0][0][1]
		
		#if i == 0:
			#print "sing mat"
			#print singular_matrix
		#	i = 1
		value = rotated[str(filename)]
		if value == 1:
			singular_matrix = zip(*singular_matrix[::-1])
		set_y.append(singular_matrix)
		
	
	#TO THEANO SHARED VARIABLE#
	set_x = numpy.asarray(set_x, dtype=theano.config.floatX)
	set_y = numpy.asarray(set_y, dtype='int32')

	set_x_len = theano.shared(set_x, borrow=True)# for the purpose of finding the length later
	set_y = theano.shared(set_y, borrow=True)

	#set_x = T.cast(set_x_len, 'float64')
	set_x = T.reshape(set_x_len, [set_x_len.shape[0], set_x_len.shape[1]*set_x_len.shape[2]])
	#set_y = T.cast(set_y,'float64') #wrong .. to keep the compiler errors short disregard when running the whole thing !!!!!!!!!!!!!
	set_y = T.reshape(set_y, [set_y.shape[0]* set_y.shape[1]*set_y.shape[2]])
	

	return set_x, set_y, set_x_len

''' MOSTLY JACKS CODE THAT WILL BE INCORPORATED LATER FOR THE PURPOSE OF SLICING THE IMAGES AND MATRICIES TO MAKE IT BOTH RUN FASTER ON THE CPU AND TO ENABLE MULTIPLE SUBSECTIONS OF THE IMAGES FOR THE PURPOSE OF MORE TRAINING DATA
def subsection(x_path_load, y_path_load, x_path_save, y_path_save):
	for folders in os.walk('x_path_load'):
		for folder in folders[1]:
			print folder
			os.chdir("x_path_load/" + folder)
			for filename in glob.glob('*.jpg'):
				img = Image.open(filename)
				img = img.crop((225, 145, 257, 177))
				img.save(filename)
			os.chdir("x_path_load")
	i=0
	for folders in os.walk('y_path_load'):
		for folder in folders[1]:
			print folder
			os.chdir("y_path_load/" + folder)
			for filename in glob.glob('*.mat'):
				mat_as_array = scipy.io.loadmat(filename)['groundTruth'][0][0][0][0][1]
				cropped_mat_as_array = mat_as_array[225:257, 145:177]
				if i==0:
					print cropped_mat_as_array
					print cropped_mat_as_array.shape
					i=1
				total_mat_file = scipy.io.loadmat(filename)
				total_mat_file['groundTruth'][0][0][0][0][0] = cropped_mat_as_array
				scipy.io.savemat(filename, total_mat_file)
			os.chdir("y_path_load")
'''
class HiddenLayer(object):
	def __init__(self, rng, input, n_in, n_out, W=None, b=None,
				 activation=T.tanh):
		"""
		Typical hidden layer of a MLP: units are fully-connected and have
		sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
		and the bias vector b is of shape (n_out,).
		NOTE : The nonlinearity used here is tanh
		Hidden unit activation is given by: tanh(dot(input,W) + b)
		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights
		:type input: theano.tensor.dmatrix
		:param input: a symbolic tensor of shape (n_examples, n_in)
		:type n_in: int
		:param n_in: dimensionality of input
		:type n_out: int
		:param n_out: number of hidden units
		:type activation: theano.Op or function
		:param activation: Non linearity to be applied in the hidden
						   layer
		"""
		self.input = input
		# end-snippet-1

		# `W` is initialized with `W_values` which is uniformely sampled
		# from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
		# for tanh activation function
		# the output of uniform if converted using asarray to dtype
		# theano.config.floatX so that the code is runable on GPU
		# Note : optimal initialization of weights is dependent on the
		#        activation function used (among other things).
		#        For example, results presented in [Xavier10] suggest that you
		#        should use 4 times larger initial weights for sigmoid
		#        compared to tanh
		#        We have no info for other function, so we use the same as
		#        tanh.
		if W is None:
			W_values = numpy.asarray(
				rng.uniform(
					low=-numpy.sqrt(6. / (n_in + n_out)),
					high=numpy.sqrt(6. / (n_in + n_out)),
					size=(n_in, n_out)
				),
				dtype=theano.config.floatX
			)
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4

			W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b

		lin_output = T.dot(input, self.W) + self.b
		self.output = (
			lin_output if activation is None
			else activation(lin_output)
		)
		# parameters of the model
		self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):
	"""Multi-Layer Perceptron Class
	A multilayer perceptron is a feedforward artificial neural network model
	that has one layer or more of hidden units and nonlinear activations.
	Intermediate layers usually have as activation function tanh or the
	sigmoid function (defined here by a ``HiddenLayer`` class)  while the
	top layer is a softmax layer (defined here by a ``LogisticRegression``
	class).
	"""

	def __init__(self, rng, input, n_in, n_hidden, n_out):
		"""Initialize the parameters for the multilayer perceptron
		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights
		:type input: theano.tensor.TensorType
		:param input: symbolic variable that describes the input of the
		architecture (one minibatch)
		:type n_in: int
		:param n_in: number of input units, the dimension of the space in
		which the datapoints lie
		:type n_hidden: int
		:param n_hidden: number of hidden units
		:type n_out: int
		:param n_out: number of output units, the dimension of the space in
		which the labels lie
		"""

		# Since we are dealing with a one hidden layer MLP, this will translate
		# into a HiddenLayer with a tanh activation function connected to the
		# LogisticRegression layer; the activation function can be replaced by
		# sigmoid or any other nonlinear function
		self.hiddenLayer = HiddenLayer(
			rng=rng,
			input=input,
			n_in=n_in,
			n_out=n_hidden,
			activation=T.tanh
		)

		# The logistic regression layer gets as input the hidden units
		# of the hidden layer
		self.logRegressionLayer = LogisticRegression(
			input=self.hiddenLayer.output,
			n_in=n_hidden,
			n_out=n_out
		)
		# end-snippet-2 start-snippet-3
		# L1 norm ; one regularization option is to enforce L1 norm to
		# be small
		self.L1 = (
			abs(self.hiddenLayer.W).sum()
			+ abs(self.logRegressionLayer.W).sum()
		)

		# square of L2 norm ; one regularization option is to enforce
		# square of L2 norm to be small
		self.L2_sqr = (
			(self.hiddenLayer.W ** 2).sum()
			+ (self.logRegressionLayer.W ** 2).sum()
		)

		# negative log likelihood of the MLP is given by the negative
		# log likelihood of the output of the model, computed in the
		# logistic regression layer
		self.negative_log_likelihood = (
			self.logRegressionLayer.negative_log_likelihood
		)
		# same holds for the function computing the number of errors
		self.errors = self.logRegressionLayer.errors

		# the parameters of the model are the parameters of the two layer it is
		# made out of
		self.params = self.hiddenLayer.params + self.logRegressionLayer.params
		# end-snippet-3


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
			 dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
	"""
	Demonstrate stochastic gradient descent optimization for a multilayer
	perceptron
	This is demonstrated on MNIST.
	:type learning_rate: float
	:param learning_rate: learning rate used (factor for the stochastic
	gradient
	:type L1_reg: float
	:param L1_reg: L1-norm's weight when added to the cost (see
	regularization)
	:type L2_reg: float
	:param L2_reg: L2-norm's weight when added to the cost (see
	regularization)
	:type n_epochs: int
	:param n_epochs: maximal number of epochs to run the optimizer
	:type dataset: string
	:param dataset: the path of the MNIST dataset file from
				 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
   """
	print "Loading data..."
	train_set_x, train_set_y, train_len = loadDataset('/home/george/Dropbox/Lab/BerkelyBenchmarkData/BSR/BSDS500/data/images/train/','/home/george/Dropbox/Lab/BerkelyBenchmarkData/BSR/BSDS500/data/groundTruth/train/')
	valid_set_x, valid_set_y, valid_len = loadDataset('/home/george/Dropbox/Lab/BerkelyBenchmarkData/BSR/BSDS500/data/images/val/','/home/george/Dropbox/Lab/BerkelyBenchmarkData/BSR/BSDS500/data/groundTruth/val/')
	test_set_x, test_set_y, test_len = loadDataset('/home/george/Dropbox/Lab/BerkelyBenchmarkData/BSR/BSDS500/data/images/test/','/home/george/Dropbox/Lab/BerkelyBenchmarkData/BSR/BSDS500/data/groundTruth/test/')
	print "...data loaded"

	# compute number of minibatches for training, validation and testing
	n_train_batches = len(train_set_x.eval()) / batch_size
	n_valid_batches = len(valid_set_x.eval()) / batch_size
	n_test_batches = len(test_set_x.eval()) / batch_size

	######################
	# BUILD ACTUAL MODEL #
	######################
	print '... building the model'

	# allocate symbolic variables for the data
	index = T.lscalar()  # index to a [mini]batch
	x = T.matrix('x')  # the data is presented as rasterized images
	y = T.ivector('y')  # the labels are presented as 1D vector of
						# [int] labels

	rng = numpy.random.RandomState(1234)

	# construct the MLP class
	classifier = MLP(
		rng=rng,
		input=x,
		n_in=481 * 321 * 3,
		n_hidden=n_hidden,
		n_out=154401
	)

	# start-snippet-4
	# the cost we minimize during training is the negative log likelihood of
	# the model plus the regularization terms (L1 and L2); cost is expressed
	# here symbolically
	cost = (
		classifier.negative_log_likelihood(y)
		+ L1_reg * classifier.L1
		+ L2_reg * classifier.L2_sqr
	)
	# end-snippet-4

	# compiling a Theano function that computes the mistakes that are made
	# by the model on a minibatch
	print "test set y"
	print test_set_y
	test_model = theano.function(
		inputs=[index],
		outputs=classifier.errors(y),
		givens={
			x: test_set_x[index * batch_size : (index + 1) * batch_size],
			y: test_set_y[index * batch_size : (index + 1) * batch_size]
		}
	)

	validate_model = theano.function(
		inputs=[index],
		outputs=classifier.errors(y),
		givens={
			x: valid_set_x[index * batch_size : (index + 1) * batch_size],
			y: valid_set_y[index * batch_size : (index + 1) * batch_size]
		}
	)

	# start-snippet-5
	# compute the gradient of cost with respect to theta (sotred in params)
	# the resulting gradients will be stored in a list gparams
	gparams = [T.grad(cost, param) for param in classifier.params]

	# specify how to update the parameters of the model as a list of
	# (variable, update expression) pairs

	# given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
	# same length, zip generates a list C of same size, where each element
	# is a pair formed from the two lists :
	#    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
	updates = [
		(param, param - learning_rate * gparam)
		for param, gparam in zip(classifier.params, gparams)
	]

	# compiling a Theano function `train_model` that returns the cost, but
	# in the same time updates the parameter of the model based on the rules
	# defined in `updates`
	train_model = theano.function(
		inputs=[index],
		outputs=cost,
		updates=updates,
		givens={
			x: train_set_x[index * batch_size : (index + 1) * batch_size],
			y: train_set_y[index * batch_size : (index + 1) * batch_size]
		}
	)
	# end-snippet-5

	###############
	# TRAIN MODEL #
	###############
	print '... training'

	# early-stopping parameters
	patience = 10000  # look as this many examples regardless
	patience_increase = 2  # wait this much longer when a new best is
						   # found
	improvement_threshold = 0.995  # a relative improvement of this much is
								   # considered significant
	validation_frequency = min(n_train_batches, patience / 2)
								  # go through this many
								  # minibatche before checking the network
								  # on the validation set; in this case we
								  # check every epoch

	best_validation_loss = numpy.inf
	best_iter = 0
	test_score = 0.
	start_time = timeit.default_timer()

	epoch = 0
	done_looping = False

	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		for minibatch_index in xrange(n_train_batches):

			minibatch_avg_cost = train_model(minibatch_index)

			# iteration number
			iter = (epoch - 1) * n_train_batches + minibatch_index

			if (iter + 1) % validation_frequency == 0:
				# compute zero-one loss on validation set
				validation_losses = [validate_model(i) for i
									 in xrange(n_valid_batches)]
				this_validation_loss = numpy.mean(validation_losses)

				print(
					'epoch %i, minibatch %i/%i, validation error %f %%' %
					(
						epoch,
						minibatch_index + 1,
						n_train_batches,
						this_validation_loss * 100.
					)
				)

				# if we got the best validation score until now
				if this_validation_loss < best_validation_loss:
					#improve patience if loss improvement is good enough
					if (
						this_validation_loss < best_validation_loss *
						improvement_threshold
					):
						patience = max(patience, iter * patience_increase)

					best_validation_loss = this_validation_loss
					best_iter = iter

					# test it on the test set
					test_losses = [test_model(i) for i
								   in xrange(n_test_batches)]
					test_score = numpy.mean(test_losses)

					print(('     epoch %i, minibatch %i/%i, test error of '
						   'best model %f %%') %
						  (epoch, minibatch_index + 1, n_train_batches,
						   test_score * 100.))
				if epoch%50:
					f = file('params.save', 'wb')
					w1_save = classifier.params[0]
					b1_save = classifier.params[1]
					w2_save = classifier.params[2]
					b2_save = classifier.params[3]

					cPickle.dump(w1_save.get_value(borrow=True), f, -1)
					cPickle.dump(b1_save.get_value(borrow=True), f, -1)
					cPickle.dump(w2_save.get_value(borrow=True), f, -1)
					cPickle.dump(b2_save.get_value(borrow=True), f, -1)
			if patience <= iter:
				done_looping = True
				break

	end_time = timeit.default_timer()
	print(('Optimization complete. Best validation score of %f %% '
		   'obtained at iteration %i, with test performance %f %%') %
		  (best_validation_loss * 100., best_iter + 1, test_score * 100.))
	print >> sys.stderr, ('The code for file ' +
						  os.path.split(__file__)[1] +
						  ' ran for %.2fm' % ((end_time - start_time) / 60.))

	f = file('params.save', 'wb')
	w1_save = classifier.params[0]
	b1_save = classifier.params[1]
	w2_save = classifier.params[2]
	b2_save = classifier.params[3]

	cPickle.dump(w1_save.get_value(borrow=True), f, -1)
	cPickle.dump(b1_save.get_value(borrow=True), f, -1)
	cPickle.dump(w2_save.get_value(borrow=True), f, -1)
	cPickle.dump(b2_save.get_value(borrow=True), f, -1)

if __name__ == '__main__':
	test_mlp()

