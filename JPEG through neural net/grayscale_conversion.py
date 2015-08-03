from PIL import Image
import PIL
import numpy as np
import cPickle
import gzip
from itertools import chain

def normalize(arr):
    min_val = np.min(arr)    
    max_val = np.max(arr)

    arr -= min_val
    arr /= (max_val - min_val)

    return arr

def subtract_mean(v):
	ray = np.mean(v)
	mean = v - ray
	return mean

img = Image.open("/Users/George/Dropbox/Lab/Neural Networks/Images/img_1.jpg")
img = img.resize((28,28), PIL.Image.ANTIALIAS)
img = img.convert("L")
img.save("resized_gscale_img.png") 
img = np.asarray(img)
arr = np.array(img)
#iterates through arr which is a list of list and turns it into one large list
one_list = list(chain(*arr))


test_set_x  = [normalize(one_list)]
test_set_y = [7]
meaningless_x = [3,4,5,6,3]
meaningless_y = 3
test_set = [test_set_x, test_set_y]
#train_set = []
#valid_set = []
f = file('mnist_test.pkl','wb')
#cPickle.dump(train_set, f, -1)
#cPickle.dump(valid_set, f, -1)
cPickle.dump(test_set, f, -1)
f.close()
f_in = open('mnist_test.pkl','rb')
f_out = gzip.open('mnist_test.pkl.gz','wb')
f_out.writelines(f_in)
f_out.close()
f_in.close()
print "picture stored"