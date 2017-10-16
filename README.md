# MNIST
Run "python MNISTNN.py" by editing the parameters at the bottom. or import the necessary modules elsehwere from MNIST.py and use the feedforwardnn() class.

net = [784, 100, 10] -> this means input layer has size 784, hidden layer has size 100, output layer has size 10
If we want to increase the number of layers, just add one more parameter to the net list that's it!
net = [784, 100, 100, 10] -> inp layer=784, two hidden layers with 100 neurons each and an output layer with 10 neurons.

network=feedforwardnn(net, '../DigitsTrain.txt','../DigitsValid.txt', '../DigitsTest.txt' )
class initialization is complete now and the class would have populated with all the data and labels as well as network architecture and weights after the init call

Next, we want to train this network, for this we just call network.train and then pass the hyperparameters that we want.

network.train(epochs=200, batch_size=10, lr=0.1, act='sigm', momentum=0.9, l2_reg=0.00001, BN='True') 
#three different activations has been implemented, one is 'Relu', 'tanh', 'sigm'
For 'Relu' and 'tanh' the good lr is 0.01 and for sigmoid the good lr is 0.1. as the names suggest, you can change different parameters and the network will train accordingly.
I didn't do gradient clipping so it for higher learning rates there may be some divergence.

That's it with two lines of code everything has been abstracted including the activation functions as well.
