# This is the implementation of the CNN with two Convpool layers
# followed by a fully connected feedforward network, vanilla neural network part

import numpy as np 
import theano
import theano.tensor as T 
import matplotlib.pyplot as plt 

# nnet.conv2d and signal.conv2d are bothh to calculate the convolution
# However:
# nnet.conv2d input and filter both are symbolic 4D tensor:
#         (batch size N, nput channels, inpit rows, input columns)
#         (output chanels, input chanels, filter rows, filter columns)
# signal.conv2d input and filter need to be 2D or 3D tensor
# What's more:
# nnet.conv2d has attribute : border_mode, which determine the size after convolution(default: valid)
#                             filter_dilation, (a,b), for dilated convolution
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

from scipy.io import loadmat
from sklearn.utils import shuffle

from datetime import datetime
from benchmark import relu, y2indicator, get_data, error_rate

def convpool(X,W,b,poolsz=(2,2)):
	conv_out = conv2d(input=X,filters=W)

	pooling_out = pool.pool_2d(
		input=conv_out,
		ws=poolsz,
		ignore_border=True
	)
	# add the bias term. Since the bias is a vector (1D array), we first
	# reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
	# thus be broadcasted across mini-batches and feature map
	# width & height
	# return T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))
	return relu(pooling_out + b.dimshuffle('x',0,'x','x'))

def init_filter(shape,poolsz):
	w = np.random.randn(*shape)/np.sqrt(np.prod(shape[1:])+shape[0]*np.prod(shape[2:]/np.prod(poolsz)))
	return w.astype(np.float32)

def rearrange(X):
	# the input should has shape(N,3,32,32) as nnet.conv2d defines, 3 means color chanel
	# the data from loadmat has shape(32,32,3,N)
	# N = X.shape[-1]
	# out = np.zeros((N,3,32,32),dtype=np.float32)
	# for n in range(N):
	#	  for i in range(3):
	#		  out[n,i,:,:] = X[:,:,i,n]
	#r eturn out/255
	return (X.transpose(3,2,0,1)/255).astype(np.float32)

def main():
	train,test = get_data()
	# matlab indexes from 1, python from 0, so need operation on y to -1
	# need to scale X by /255
	# need indicator matrix

	# 1. set(train)
	#    {'X', '__globals__', '__header__', '__version__', 'y'}
	# 2. train['X'].shape
	#    (32, 32, 3, 73257):(32,32,3,N)
	# 3. train['y'].shape
	#    (73257, 1)
	# 4.  train['y'] is array of list of 1 dim list, flatten() transform into array of list
	Xtrain = rearrange(train['X'])
	Ttrain = train['y'].flatten()-1
	del train
	Xtrain,Ttrain = shuffle(Xtrain,Ttrain)
	Ttrain_ind = y2indicator(Ttrain)

	Xtest = rearrange(test['X'])
	Ttest = test['y'].flatten()-1
	del test
	Ttest_ind = y2indicator(Ttest)

	max_iter = 8
	print_period = 10
	
	N = Xtrain.shape[0]
	batch_sz = 500
	n_batches = N // batch_sz

	lr = np.float32(0.00001)
	reg = np.float32(0.01)
	mu = np.float32(0.99) #use the momentum GD

	M = 500
	K = 10
	poolsz = (2,2)

	# step1: set initials for W and b, which are tensor.shared
	W1_sahpe = (20,3,5,5)
	W1_init = init_filter(W1_sahpe,poolsz)
	b1_init = np.zeros(W1_sahpe[0],dtype=np.float32)
	# after Conv: dim = 32 - 5 + 1 = 28 (for border_mode = valid, a default value)
	# after pooling: dim = 28/2 = 14

	W2_sahpe = (50,20,5,5)
	W2_init = init_filter(W2_sahpe,poolsz)
	b2_init = np.zeros(W2_sahpe[0],dtype=np.float32)
	# after Conv: dim = 14 - 5 + 1 = 10 (for border_mode = valid, a default value)
	# after pooling: dim = 10/2 = 5

	# for the vanilla network part
	W3_init = np.random.randn(W2_sahpe[0]*5*5,M)/np.sqrt(W2_sahpe[0]*5*5+M)
	b3_init = np.zeros(M,dtype=np.float32)
	W4_init = np.random.randn(M,K)/np.sqrt(M+K)
	b4_init = np.zeros(K,dtype=np.float32)


	# step2: Define the theano variables and theano expression
	X = T.tensor4('X',dtype='float32')
	Y = T.matrix('Y')
	
	W1 = theano.shared(W1_init, 'W1')
	b1 = theano.shared(b1_init, 'b1')
	W2 = theano.shared(W2_init, 'W2')
	b2 = theano.shared(b2_init, 'b2')
	W3 = theano.shared(W3_init.astype(np.float32), 'W3')
	b3 = theano.shared(b3_init, 'b3')
	W4 = theano.shared(W4_init.astype(np.float32), 'W4')
	b4 = theano.shared(b4_init, 'b4')

	# momentum changes
	dW1 = theano.shared(np.zeros(W1_init.shape,dtype=np.float32),'dW1')
	db1 = theano.shared(np.zeros(b1_init.shape,dtype=np.float32),'db1')
	dW2 = theano.shared(np.zeros(W2_init.shape, dtype=np.float32), 'dW2')
	db2 = theano.shared(np.zeros(b2_init.shape, dtype=np.float32), 'db2')
	dW3 = theano.shared(np.zeros(W3_init.shape, dtype=np.float32), 'dW3')
	db3 = theano.shared(np.zeros(b3_init.shape, dtype=np.float32), 'db3')
	dW4 = theano.shared(np.zeros(W4_init.shape, dtype=np.float32), 'dW4')
	db4 = theano.shared(np.zeros(b4_init.shape, dtype=np.float32), 'db4')

	# forward process
	Z1 = convpool(X,W1,b1)
	Z2 = convpool(Z1,W2,b2)
	Z3 = relu(Z2.flatten(ndim=2).dot(W3)+b3)
	pY = T.nnet.softmax(Z3.dot(W4)+b4)

	# cost expresion and prediction expression
	params = [W1,b1,W2,b2,W3,b3,W4,b4]
	rcost = reg*np.sum((p*p).sum() for p in params)
	cost = -(Y*T.log(pY)).sum()+rcost
	prediction = T.argmax(pY,axis=1)

	# step 3: train function and cost_predict function
	# updates
	update_W1 = W1 + mu*dW1 - lr*T.grad(cost, W1)
	update_b1 = b1 + mu*db1 - lr*T.grad(cost, b1)
	update_W2 = W2 + mu*dW2 - lr*T.grad(cost, W2)
	update_b2 = b2 + mu*db2 - lr*T.grad(cost, b2)
	update_W3 = W3 + mu*dW3 - lr*T.grad(cost, W3)
	update_b3 = b3 + mu*db3 - lr*T.grad(cost, b3)
	update_W4 = W4 + mu*dW4 - lr*T.grad(cost, W4)
	update_b4 = b4 + mu*db4 - lr*T.grad(cost, b4)

	update_dW1 = mu*dW1 - lr*T.grad(cost, W1)
	update_db1 = mu*db1 - lr*T.grad(cost, b1)
	update_dW2 = mu*dW2 - lr*T.grad(cost, W2)
	update_db2 = mu*db2 - lr*T.grad(cost, b2)
	update_dW3 = mu*dW3 - lr*T.grad(cost, W3)
	update_db3 = mu*db3 - lr*T.grad(cost, b3)
	update_dW4 = mu*dW4 - lr*T.grad(cost, W4)
	update_db4 = mu*db4 - lr*T.grad(cost, b4)

	updates = [
		(W1, update_W1),
		(b1, update_b1),
		(W2, update_W2),
		(b2, update_b2),
		(W3, update_W3),
		(b3, update_b3),
		(W4, update_W4),
		(b4, update_b4),
		(dW1, update_dW1),
		(db1, update_db1),
		(dW2, update_dW2),
		(db2, update_db2),
		(dW3, update_dW3),
		(db3, update_db3),
		(dW4, update_dW4),
		(db4, update_db4),
	]

	train = theano.function(
		inputs=[X,Y],
		updates = updates
	)
	cost_predict = theano.function(
		inputs=[X,Y],
		outputs=[cost,prediction])

	# step4: combine data and theano structure
	t0 = datetime.now()
	costs=[]
	for i in range(max_iter):
		for n in range(n_batches):
			Xbatch = Xtrain[n*batch_sz:n*batch_sz+batch_sz]
			Tbatch = Ttrain_ind[n*batch_sz:n*batch_sz+batch_sz]

			train(Xbatch,Tbatch)
			if n%print_period== 0:
				c,p = cost_predict(Xtest,Ttest_ind)
				err = error_rate(Ttest,p)
				costs.append(c)
				print('cost/err at iteration i=%d n=%d is %.3f/%.3f'%(i,n,c,err))
	print("Elapsed time:", (datetime.now() - t0))
	plt.plot(costs)
	plt.show()

if __name__ == '__main__':
	main()

