import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from copy import deepcopy

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def dsigmoid(x):
	return sigmoid(x)*(1-sigmoid(x))

def relu(x):
	return np.maximum(x, 0, x)

def drelu(x):
	dx = deepcopy(x)
	dx[x<=0] = 0
	return dx

def tanh(x):
	return np.tanh(x)

def dtanh(x):
	return 1.0 - np.tanh(x)**2


def softmax(x):
    return np.exp(x- np.max(x)) / np.sum(np.exp(x- np.max(x)), axis=0)

def ceerror(x, y):
	l = -np.log(x+0.0000001)
	return np.mean(np.sum(l*y, axis=0))




class datareader():
	def __init__(self, file_path):
		self.path=file_path

	def returndata(self):
		self.file = open(self.path, "r")
		strdata = self.file.readlines()
		self.data=np.zeros((len(strdata), 784), dtype=np.float)
		self.labels=np.zeros((len(strdata), 10), dtype=np.int)
		for i in range(len(strdata)):
			data = strdata[i].split(',')
			for j in range(784):
				self.data[i,j]=float(data[j])
			self.labels[i,int(data[784])] = 1
		#TOFO return the images by scanning line by line
		return self.data, self.labels

#We will initialize the weights of the netwrok here
def init_weights(net, BN):
	params = {}
	for i in range(len(net)-1):
		params[i] = {}
		n = net[i]
		m = net[i+1]
		s = np.sqrt(3./n)
		params[i]['w'] = s*np.random.randn(n, m)
		params[i]['b'] = np.zeros((m,1))
		if BN=='True':
			params[i]['gamma'] = np.ones((m,1))
			params[i]['beta'] = np.zeros((m,1))
			params[i]['gvar'] = np.ones((m,1))
			params[i]['gmean'] = np.zeros((m,1))
	return params

#We will make the grad parameters go to zero before every backward pass
def init_grads(net, BN):
	gradparams = {}
	for i in range(len(net)-1):
		gradparams[i] = {}
		n = net[i]
		m = net[i+1]
		gradparams[i]['w'] = np.zeros((n,m))
		gradparams[i]['b'] = np.zeros((m,1))
		if BN=='True':
			gradparams[i]['gamma'] = np.zeros((m,1))
			gradparams[i]['beta'] = np.zeros((m,1))
	return gradparams

def ip_forward(inp, param, s, mode, BN='False'):
	# input is nxk where n is the dimensionaltiy and k is the batch size
	# output is mxk where m is the output dimensionality
	# param{w} has the weights for this particular layer
	# param{b} has the biases for this particular layer
	# out = (w.T)X + b
	n,k = inp.shape
	n,m = param['w'].shape
	out = np.zeros((m, k), dtype=np.float)
	b = np.tile(param['b'], (1, k))

	param['a'] = param['w'].T.dot(inp) + b
	out = deepcopy(param['a'])
	if BN=='True':
		param['bn'], param['cache'], param['gmean'], param['gvar'] = bn_forward(param['a'],param['gamma'], param['beta'], 0.9,param['gmean'], param['gvar'], mode )
		out = deepcopy(param['bn'])
	if(s=='sigm'):
		param['h'] = sigmoid(out)
	if(s=='Relu'):
		param['h'] = relu(out)
	if(s=='tanh'):
		param['h'] = tanh(out)
	if(s=='soft'):
		param['h'] = softmax(param['a'])
	return param['h']

def bn_forward(x, gamma, beta, m, mean, var, mode='Train'):
	if mode=='Train':

		k, n= x.shape
		mu = (1./n)*(np.expand_dims(np.sum(x, axis=1), axis=1))
		mean = m*mean + (1-m)*mu
		x_mu = x-mu
		s = x_mu ** 2
		var = 1./n * np.expand_dims(np.sum(s, axis=1), axis=1)
		var = m*var + (1-m)*var
		sqvar = np.sqrt(var + 0.0000001)
		ivar = 1./sqvar
		x_hat = x_mu*ivar
		gamma_x = gamma*x_hat
		out = gamma_x + beta 
		data = (x_hat, gamma, x_mu, ivar, sqvar, var)
	else:
		#This implementation should contain the test behavior where we use the global mean and varinace
		xhat = (x - mean) / np.sqrt(var + 0.0000001)
		out = gamma * xhat + beta
		data = (mean, var, gamma, beta)
	return out, data, mean, var

def bn_backward(dy, cache, j):
	x_hat, gamma, x_mu, ivar, sqvar, var = cache
	k, n = dy.shape
	dbeta = np.sum(dy, axis=1)
	dgamma = np.sum(dy*x_hat, axis=1)
	dx_hat = dy*gamma
	divar = np.expand_dims(np.sum(dx_hat*x_mu, axis=1), axis=1)
	a = (sqvar**2)
	dsqvar = (-1./(sqvar**2))*divar
	dvar = 0.5*(1./np.sqrt(var+0.0000001))*dsqvar
	dsq = (1./n)*(np.ones((k, n)))*dvar
	dx_mu1 = dx_hat*ivar 
	dx_mu2 = 2 *np.expand_dims(x_mu[:,j], axis=1)* dsq
	dx1 = (dx_mu1 + dx_mu2)
	dmu = -1 * (dx_mu1+dx_mu2)
	dx2 = 1. /n * np.ones((k, n)) * dmu
	dx = dx1 + dx2
	return dx, np.expand_dims(dgamma, axis=1), np.expand_dims(dbeta, axis=1)




class feedforwardnn():
	def __init__(self, net, trainpath, validpath, testpath):
		d=datareader(trainpath)
		self.Traindata, self.Trainlabels = d.returndata()
		a,b=self.Traindata.shape
		self.num = a
		self.net = net
		self.ind = [i for i in range(self.num)]
		v=datareader(validpath)
		self.Validdata, self.Validlabels = v.returndata()
		v=datareader(testpath)
		self.Testdata, self.Testlabels = v.returndata()
		self.trainloss=[]
		self.trainp=[]
		self.validloss=[]
		self.validp=[]
		self.testloss=[]
		self.testp=[]

		self.validloss
	def init_net(self, net):
		self.p= init_weights(net, self.BN)
		return self.p
	def fp(self, inp):
		temp = deepcopy(inp)
		for i in range(len(net)-2):
			temp = ip_forward(temp, self.p[i], self.act, self.mode, self.BN)
		temp = ip_forward(temp, self.p[len(net)-2], 'soft', 'Train')
		return temp
	def backward(self, inp, exp, batch_size, lr, m=None, l2_reg=None):
		#This function will do the backward pass and return the gradients
		self.batch = batch_size
		if(m==None): #No need to store previous gradients if momentum is not used
			self.gradp = init_grads(net, self.BN)
			prev_grad = deepcopy(self.gradp)
			m = 0
		else:
			prev_grad = deepcopy(self.gradp)
			self.gradp = init_grads(net, self.BN)
		for i in range(batch_size):
			error = np.expand_dims(self.p[len(net)-2]['h'][:,i] - exp[:,i], axis=1)
			#print(error.shape)
			#print(self.p[len(net)-3]['h'].shape)
			self.gradp[len(net)-2]['w'] = self.gradp[len(net)-2]['w'] + np.dot(np.expand_dims(self.p[len(net)-3]['h'][:,i],axis=1), error.T)
			self.gradp[len(net)-2]['b'] = self.gradp[len(net)-2]['b'] + error
			error = np.dot(self.p[len(net)-2]['w'], error)
			j=len(net)-3
			if self.act=='sigm':
				while(j>0):
					error = (error*np.expand_dims(dsigmoid(self.p[j]['a'][:,i]), axis=1))
					if self.BN == 'True':
						error, dgamma, dbeta = bn_backward(error, self.p[j]['cache'], i)
						self.gradp[j]['beta'] = self.gradp[j]['beta'] + dbeta
						self.gradp[j]['gamma'] = self.gradp[j]['gamma'] + dgamma
					#BN's diff to follow ere if present and the error changes accordingly
					self.gradp[j]['w'] = self.gradp[j]['w'] + np.dot(np.expand_dims(self.p[j-1]['h'][:,i],axis=1), error.T)
					self.gradp[j]['b'] = self.gradp[j]['b']	+ error
					error = np.dot(self.p[j]['w'], error)	
					j = j-1
				error = (error*np.expand_dims(dsigmoid(self.p[0]['a'][:,i]), axis=1))
				if self.BN == 'True':
					error, dgamma, dbeta = bn_backward(error, self.p[0]['cache'], i)
					self.gradp[0]['beta'] = self.gradp[0]['beta'] + dbeta
					self.gradp[0]['gamma'] = self.gradp[0]['gamma'] + dgamma
				self.gradp[0]['w'] = self.gradp[0]['w'] + np.dot(np.expand_dims(inp[:,i],axis=1), error.T)
				self.gradp[0]['b'] = self.gradp[0]['b'] + error 
			elif self.act=='Relu':
				while(j>0):
					error = (error*np.expand_dims(drelu(self.p[j]['a'][:,i]), axis=1))
					if self.BN == True:
						error, dgamma, dbeta = bn_backward(error, self.p[j]['cache'], i)
						self.gradp[j]['beta'] = self.gradp[j]['beta'] + dbeta
						self.gradp[j]['gamma'] = self.gradp[j]['gamma'] + dgamma
					#BN's diff to follow ere if present and the error changes accordingly
					self.gradp[j]['w'] = self.gradp[j]['w'] + np.dot(np.expand_dims(self.p[j-1]['h'][:,i],axis=1), error.T)
					self.gradp[j]['b'] = self.gradp[j]['b']	+ error
					error = np.dot(self.p[j]['w'], error)	
					j = j-1
				error = (error*np.expand_dims(drelu(self.p[0]['a'][:,i]), axis=1))
				if self.BN == 'True':
					error, dgamma, dbeta = bn_backward(error, self.p[0]['cache'], i)
					self.gradp[0]['beta'] = self.gradp[0]['beta'] + dbeta
					self.gradp[0]['gamma'] = self.gradp[0]['gamma'] + dgamma
				self.gradp[0]['w'] = self.gradp[0]['w'] + np.dot(np.expand_dims(inp[:,i],axis=1), error.T)
				self.gradp[0]['b'] = self.gradp[0]['b'] + error 
			elif self.act=='tanh':
				while(j>0):
					error = (error*np.expand_dims(dtanh(self.p[j]['a'][:,i]), axis=1))
					if self.BN == True:
						error, dgamma, dbeta = bn_backward(error, self.p[j]['cache'])
						self.gradp[j]['beta'] = self.gradp[j]['beta'] + dbeta
						self.gradp[j]['gamma'] = self.gradp[j]['gamma'] + dgamma
					self.gradp[j]['w'] = self.gradp[j]['w'] + np.dot(np.expand_dims(self.p[j-1]['h'][:,i],axis=1), error.T)
					self.gradp[j]['b'] = self.gradp[j]['b']	+ error
					error = np.dot(self.p[j]['w'], error)	
					j = j-1
				error = (error*np.expand_dims(dtanh(self.p[0]['a'][:,i]), axis=1))
				if self.BN == 'True':
					error, dgamma, dbeta = bn_backward(error, self.p[0]['cache'], i)
					self.gradp[0]['beta'] = self.gradp[0]['beta'] + dbeta
					self.gradp[0]['gamma'] = self.gradp[0]['gamma'] + dgamma
				self.gradp[0]['w'] = self.gradp[0]['w'] + np.dot(np.expand_dims(inp[:,i],axis=1), error.T)
				self.gradp[0]['b'] = self.gradp[0]['b'] + error 
			for j in range(len(net)-1):
				if l2_reg == None:
					self.gradp[j]['w'] = m*prev_grad[j]['w'] + (lr/batch_size)*(self.gradp[j]['w'])
				else:
					self.gradp[j]['w'] = m*prev_grad[j]['w'] + (lr/batch_size)*(self.gradp[j]['w'] + l2_reg*self.p[j]['w'])
				self.gradp[j]['b'] = (lr/batch_size)*(self.gradp[j]['b'])
				if self.BN == 'True':
					self.gradp[j]['gamma'] = (lr/batch_size)*(self.gradp[j]['gamma'])
					self.gradp[j]['beta'] = (lr/batch_size)*(self.gradp[j]['beta'])
			return self.gradp
	def update_weights(self):
		#This function will update the weights
		for i in range(len(self.net)-1):
				self.p[i]['w'] = self.p[i]['w'] - self.gradp[i]['w']
				self.p[i]['b'] = self.p[i]['b'] - self.gradp[i]['b']
				if self.BN=='True':
					self.p[i]['gamma'] = self.p[i]['gamma'] - self.gradp[i]['gamma']
					self.p[i]['beta'] = self.p[i]['beta'] - self.gradp[i]['beta']
	def train(self, epochs, batch_size, lr, act='sigm',momentum=None, l2_reg=None, BN='False'):
		self.BN = BN
		self.init_net(self.net)
		self.gradp = init_grads(net, self.BN)
		self.act = act

		iters = (self.num/batch_size)*epochs
		for i in range(iters):
			self.mode = 'Train'
			ind = np.random.choice(self.ind, batch_size, replace=False)
			xtrain = np.vstack([self.Traindata[j,:] for j in ind]).T
			ytrain = np.vstack([self.Trainlabels[j,:] for j in ind]).T
			o = self.fp(xtrain) #Forward pass through the network
			self.backward(xtrain, ytrain, batch_size, lr, momentum, l2_reg)
			self.update_weights()
			if((i*batch_size)%self.num ==0):
				print(i)
				#accuracy on all data
				self.trainacc()
				self.mode = 'Test'
				self.validacc()
				self.testacc()
		#self.plotweights()
		self.plotloss()
		#This function trains the network
	def plotweights(self):
		fig, axes = plt.subplots(22,10)
		w = self.p[0]['w']
		#w = np.reshape(w,(28,28,100))
		for i in range(22):
			for j in range(10):
				axes[i,j].matshow(w[:,10*i+j].reshape(28, 28), cmap=plt.cm.gray)
				axes[i,j].set_xticks(())
				axes[i,j].set_yticks(())
		plt.show()
		

		#plt.tight_layout() # do not use this!!
		plt.show()
	def plotloss(self):
		plt.plot(self.trainloss)
		plt.plot(self.validloss)
		plt.plot(self.testloss)
		plt.legend(['TrainLoss', 'ValidLoss', 'TestLoss'], loc='upper left')
		plt.show()
		plt.plot(self.trainp)
		plt.plot(self.validp)
		plt.plot(self.testp)
		plt.legend(['TrainMisClassification', 'ValidMisCalssification', 'TestMisCalssification'], loc='bottom right')
		plt.show()
	def validacc(self):
		#This function will be used to validate the data
		out = self.fp(self.Validdata.T)
		self.validloss.append(ceerror(out, self.Validlabels.T))
		pred = np.argmax(out, axis=0)
		labels = np.argmax(self.Validlabels, axis=1)
		numc = np.sum(pred == labels)
		perc = 1-(numc*1.0/pred.shape[0])
		self.validp.append(perc)
		print("The percentage of incorrect predictions in validation data is ", perc)
	def testacc(self):
		#This function will be used to validate the data
		out = self.fp(self.Testdata.T)
		self.testloss.append(ceerror(out, self.Testlabels.T))
		pred = np.argmax(out, axis=0)
		labels = np.argmax(self.Testlabels, axis=1)
		numc = np.sum(pred == labels)
		perc = 1-(numc*1.0/pred.shape[0])
		self.testp.append(perc)
		print("The percentage of incorrect predictions in test data is ", perc)
	def trainacc(self):
		#This function will be used to validate the data
		out = self.fp(self.Traindata.T)
		self.trainloss.append(ceerror(out, self.Trainlabels.T))
		pred = np.argmax(out, axis=0)
		labels = np.argmax(self.Trainlabels, axis=1)
		numc = np.sum(pred == labels)
		perc = 1-(numc*1.0/pred.shape[0])
		self.trainp.append(perc)
		print("The percentage of incorrect predictions  in training data is ", perc)






		


#This defines out network architecture
net = [784, 100, 10]
batch_size = 10

network=feedforwardnn(net, '../../DigitsTrain.txt','../../DigitsValid.txt', '../../DigitsTest.txt' )
network.train(epochs=200, batch_size=10, lr=0.1, act='sigm', momentum=0, l2_reg=0, BN='False')






#out = t.fp()


