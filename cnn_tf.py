import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 

from sklearn.utils import shuffle
from scipy.signal import convolve2d
from scipy.io import loadmat
from datetime import datetime

from benchmark import get_data, y2indicator, error_rate

def convpool(X,W,b):
	# this is convolve and pooling function
	# input X (batch,input_height,input_width,input_chanels)
	# filter W: (filter_height,filter_width,input_chanels,output_chanels)
	# output[b, i, j, :] =
    # sum_{di, dj} (input[b, strides[1]*i+di-pad_top, strides[2]*j+dj-pad_left, ...] *filter[di, dj, ...])
    # above sum funion is sum of matrix multiplication, each is (input chanel dim)*(in_dim,out_dim)
    # pad_top, pad_left are determined by padding type, here 'SAME' means no change of shape
    # above, ezch convpool, shape(a,b)-> shape (a/2,b/2)
	conv_out = tf.nn.conv2d(X,W,strides=[1,1,1,1],padding='SAME')
	conv_out = tf.nn.bias_add(conv_out,b)
	pool_out = tf.nn.max_pool(conv_out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	return tf.nn.relu(pool_out)

def rearrange(X):
	# in thesorflow, the input are 4D tensor:(batch,in_hieght,in_width,in_chanel)
	return (X.transpose(3,0,1,2)/255).astype(np.float32)

def init_weight(shape,poolsz):
	w = np.random.randn(*shape)/np.sqrt(np.prod(shape[:-1])+shape[-1]*np.prod(shape[:-2]/np.prod(poolsz)))
	return w.astype(np.float32)

def main():
	# step 1: get data and do some transformation, datatype, shape, value range, ....
	train,test = get_data()
	
	Xtrain = rearrange(train['X'])
	Ttrain = train['y'].flatten()-1
	del train
	Xtrain, Ttrain = shuffle(Xtrain,Ttrain)
	Ttrain_ind = y2indicator(Ttrain)

	Xtest = rearrange(test['X'])
	Ttest = test['y'].flatten()-1
	del test
	Ttest_ind = y2indicator(Ttest)

	# step 2: parameter initialize
	max_iter = 7
	print_period = 10
	N = Xtrain.shape[0]
	batch_sz = 500
	n_batches = N // batch_sz

    # limit samples since input will always have to be same size
    # you could also just do N = N / batch_sz * batch_sz
	Xtrain = Xtrain[:73000,]
	Ttrain = Ttrain[:73000]
	Xtest = Xtest[:26000,]
	Ttest = Ttest[:26000]
	Ttest_ind = Ttest_ind[:26000,]

	M = 500
	K = 10
	poolsz=[2,2]

	# shape of filters in Convpool part
	W1_shape = [5,5,3,20]
	W2_shape = [5,5,20,50]
	# Convpool part initialization
	W1_init = init_weight(W1_shape,poolsz)
	b1_init = np.zeros(W1_shape[-1],dtype=np.float32)
	W2_init = init_weight(W2_shape,poolsz)
	b2_init = np.zeros(W2_shape[-1],dtype=np.float32)
	# vanila net part initialization, after convpool stage, each feature map has(32/2/2=8) 8*8features
	W3_init = np.random.randn(W2_shape[-1]*8*8,M)/np.sqrt(W2_shape[-1]*8*8+M)
	b3_init = np.zeros(M,dtype=np.float32)
	W4_init = np.random.randn(M,K)/np.sqrt(M+K)
	b4_init = np.zeros(K,dtype=np.float32)

	#step 3: structure in tensorflow
	X = tf.placeholder(tf.float32,shape=(batch_sz,32,32,3),name='X')
	T = tf.placeholder(tf.float32,shape=(batch_sz,K),name='T')
	W1 = tf.Variable(W1_init.astype(np.float32))
	b1 = tf.Variable(b1_init.astype(np.float32))
	W2 = tf.Variable(W2_init.astype(np.float32))
	b2 = tf.Variable(b2_init.astype(np.float32))
	W3 = tf.Variable(W3_init.astype(np.float32))
	b3 = tf.Variable(b3_init.astype(np.float32))
	W4 = tf.Variable(W4_init.astype(np.float32))
	b4 = tf.Variable(b4_init.astype(np.float32))	

	Z1 = convpool(X,W1,b1)
	Z2 = convpool(Z1,W2,b2)
	Z2_shape = Z2.get_shape().as_list()
	Z2_re = tf.reshape(Z2,[Z2_shape[0],np.prod(Z2_shape[1:])])
	Z3 = tf.nn.relu(tf.matmul(Z2_re,W3)+b3)
	Z4 = tf.matmul(Z3,W4)+b4

	# cost function and train, predict operation
	cost = tf.reduce_sum(
		tf.nn.softmax_cross_entropy_with_logits(
			logits=Z4,
			labels=T
		)
	)

	train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)
	predict_op = tf.argmax(Z4,1)

	# step4: data combine with tensorflow structure
	costs = []
	t0 = datetime.now()
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)

		for i in range(max_iter):
			for j in range(n_batches):
				Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz),]
				Ybatch = Ttrain_ind[j*batch_sz:(j*batch_sz+batch_sz),]

				if len(Xbatch)==batch_sz:
					session.run(train_op,feed_dict={X:Xbatch,T:Ybatch})
					if j%print_period==0:
						# due to RAM limitations we need to have a fixed size input
						# so as a result, we have this ugly total cost and prediction computation
						c = 0
						p = np.zeros(len(Xtest))
						for n in range(len(Xtest)//batch_sz):
							Xtestbatch = Xtest[n*batch_sz:(n*batch_sz+batch_sz),]
							Ytetsbatch = Ttest_ind[n*batch_sz:(n*batch_sz+batch_sz),]
							c += session.run(cost,feed_dict={X:Xtestbatch,T:Ytetsbatch})
							p[n*batch_sz:(n*batch_sz+batch_sz)] = session.run(
								predict_op,feed_dict={X:Xtestbatch}
							)
						err = error_rate(p,Ttest)
						costs.append(c)
						print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, c, err))
	print("Elapsed time:", (datetime.now() - t0))
	plt.plot(costs)
	plt.show()

if __name__ == '__main__':
	main()


























