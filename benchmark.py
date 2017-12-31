import os
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

from scipy.io import loadmat
from sklearn.utils import shuffle
from datetime import datetime

def y2indicator(y):
    N = len(y)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def error_rate(p, t):
    return np.mean(p != t)

def flatten(X):
	# input shape:(32,32,3,N), 3 is 3 dimension for the color variable
 	# output shape:(N,3072), 3072 = 32x32x3
 	N = X.shape[-1]
 	flat = np.zeros([N,3072])
 	for n in range(N):
 		flat[n] = X[:,:,:,n].reshape(3072)
 	return flat

def relu(a):
	return a * (a>0)

def get_data():
	if not os.path.exists('/Users/zehaodong/research/street_view_house_numbers/large_files/train_32x32.mat'):
		print('There is no traininf dataset existing')
		print('Please get the data from: https:http://ufldl.stanford.edu/housenumbers')
		print('Place train_32x32.mat in the folder large_files adjacent to the class folder')
		exit()

	train = loadmat('/Users/zehaodong/research/street_view_house_numbers/large_files/train_32x32.mat')
	test = loadmat('/Users/zehaodong/research/street_view_house_numbers/large_files/test_32x32.mat')
	return train, test

def weigths_and_bias_init(M1,M2):
	W = np.random .randn(M1,M2)/np.sqrt(M1+M2)
	b = np.zeros(M2)
	return W.astype(np.float32),b.astype(np.float32)


def main():
	train,test = get_data()
	# matlab indexes from 1, python from 0, so need operation on y to -1
	# need to scale X by /255
	# need indicator matrix
	Xtrain = flatten(train['X'].astype(np.float32)/255)
	Ytrain = train['y'].flatten()-1
	# 1. set(train)
	#    {'X', '__globals__', '__header__', '__version__', 'y'}
	# 2. train['X'].shape
	#    (32, 32, 3, 73257)
	# 3. train['y'].shape
	#    (73257, 1)
	# 4.  train['y'] is array of list of 1 dim list, flatten() transform into array of list

	Xtrain,Ytrain = shuffle(Xtrain,Ytrain)
	Ytrain_ind = y2indicator(Ytrain)

	Xtest  = flatten(test['X'].astype(np.float32) / 255)
	Ytest  = test['y'].flatten() - 1
	Ytest_ind  = y2indicator(Ytest)

	# gradient descent params
	max_iter = 20
	print_period = 10
	N, D = Xtrain.shape
	batch_sz = 500
	n_batches = int(N / batch_sz)

	# initial weights
	M1 = 1000
	M2 =500  
	K =10
	W1_init,b1_init = weigths_and_bias_init(D,M1)
	W2_init,b2_init = weigths_and_bias_init(M1,M2)
	W3_init,b3_init = weigths_and_bias_init(M2,K)

	# define variables and expressions
	X = tf.placeholder(tf.float32, shape=(None, D), name='X')
	T = tf.placeholder(tf.float32, shape=(None, K), name='T')
	W1 = tf.Variable(W1_init)
	b1 = tf.Variable(b1_init)
	W2 = tf.Variable(W2_init)
	b2 = tf.Variable(b2_init)
	W3 = tf.Variable(W3_init)
	b3 = tf.Variable(b3_init)

	Z1 = tf.nn.relu( tf.matmul(X, W1) + b1 )
	Z2 = tf.nn.relu( tf.matmul(Z1, W2) + b2 )
	Yish = tf.matmul(Z2, W3) + b3

	cost = tf.reduce_sum(
		tf.nn.softmax_cross_entropy_with_logits(logits=Yish, labels=T)
	)

	train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)

	# we'll use this to calculate the error rate
	predict_op = tf.argmax(Yish, 1)

	t0 = datetime.now()
	LL = []
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)

		for i in range(max_iter):
			for j in range(n_batches):
				Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
				Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]

				session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
				if j % print_period == 0:
					test_cost = session.run(cost, feed_dict={X: Xtest, T: Ytest_ind})
					prediction = session.run(predict_op, feed_dict={X: Xtest})
					err = error_rate(prediction, Ytest)
					print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
					LL.append(test_cost)
	print("Elapsed time:", (datetime.now() - t0))
	plt.plot(LL)
	plt.show()




if __name__ == '__main__':
	main()














