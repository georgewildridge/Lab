def loadDataset():
   train_dataset = scipy.io.loadmat("train_32x32.mat")
   train_set_x = []
   for i in range(0, 50000):
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

   return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y