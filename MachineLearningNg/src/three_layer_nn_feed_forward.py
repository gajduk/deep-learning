import theano
import numpy
import pylab
import theano.tensor as T
rng = numpy.random

#--------------------
#set some parameters
#--------------------
#learning rate
alpha = 2
max_iterations = 2000
#number of classes
k = 10
#number of hidden units
n_hidden = 25
#epsilon, range of initial theta gueses ( e = sqrt(6)/(sqrt(n_output+n_input)) )
e = 0.12
#regularization parameter
l = 0.0001
#--------------------
#load the data
#--------------------
from numpy import genfromtxt
my_data = genfromtxt('ml3data1.csv', delimiter=',')
m,n = my_data.shape
#features and labels
xn = my_data[:,0:n-1]
yn = my_data[:,n-1]
#add bias features
#input_x = numpy.ones((m,n))
#input_x[:,1:n] = xn
input_x = xn
#split the data in separate train and test sets
test_idxs = rng.randint(0,m,1120)
train_idxs = numpy.setdiff1d(numpy.arange(0,m,1),test_idxs).astype(int)
test_idxs = numpy.setdiff1d(numpy.arange(0,m,1),train_idxs).astype(int)
test_xn = input_x[test_idxs,:]
test_yn = yn[test_idxs].astype(int)
train_xn = input_x[train_idxs,:]
train_yn = yn[train_idxs].astype(int)
train_m = train_xn.shape[0]
test_m = test_xn.shape[0]
#cast train_yn for Ng type learning
input_y = numpy.zeros((train_m,k))
input_y[range(train_m),train_yn] = 1
input_y = input_y.astype(int)

#--------------------
#declare theano variables
#--------------------
X = theano.tensor.matrix('X', dtype='floatX')
Y = theano.tensor.matrix('Y', dtype='int32')
#level1 variables
theta1 = theano.shared( rng.randn(n-1,n_hidden)*2*e-e, name="theta1",borrow=True)
theta1_bias = theano.shared( rng.randn(1,n_hidden).reshape(1,n_hidden)*2*e-e, name="theta1_bias", borrow=True, broadcastable=(True,False))
#level 2 variables
theta2 = theano.shared(rng.randn(n_hidden,k)*2*e-e, name="theta2",borrow=True)
theta2_bias = theano.shared(rng.randn(1,k).reshape(1,k)*2*e-e, name="theta2_bias",borrow=True, broadcastable=(True,False))

#--------------------
#feed forward activation function
#--------------------
a2 = T.nnet.sigmoid(T.dot(X,theta1)+theta1_bias)
a3 = T.nnet.sigmoid(T.dot(a2,theta2)+theta2_bias)

predict_y = T.argmax(a3,axis=1)

#cost function
cost = 1.0/train_m*T.sum(-Y*T.log(a3)-(1-Y)*(T.log(1-a3)))

regularized_cost = cost+l/2/train_m*(T.sum(theta1**2)+T.sum(theta2**2)+T.sum(theta1_bias**2)+T.sum(theta2_bias**2))

#gradients
gtheta1,gtheta2,gtheta1_bias,gtheta2_bias = T.grad(regularized_cost,(theta1,theta2,theta1_bias,theta2_bias))

#error functions
#d3 = a3-Y

#d2 = T.dot(d3,theta2)*g_prim_z2

#D2 = T.outer(T.sum(d3,0),a2)

#D1 = T.outer(T.sum(d2,0),X)

train = theano.function(inputs=[X,Y],outputs=[regularized_cost],updates=((theta1,theta1-alpha*gtheta1),(theta2,theta2-alpha*gtheta2),
                        (theta1_bias,theta1_bias-alpha*gtheta1_bias),(theta2_bias,theta2_bias-alpha*gtheta2_bias)),allow_input_downcast=True)    
                        
predict = theano.function(inputs=[X,],outputs=[predict_y,])

costs = []
train_accs = []
test_accs = []
for i in range(max_iterations):
    costs.append(train(train_xn,input_y))
    if i % 100 == 0:
        print "Iteration:",i
        print "Train accuracy:",numpy.sum(predict(train_xn)==train_yn)*1.0/train_m
        print "Test accuracy:",numpy.sum(predict(test_xn)==test_yn)*1.0/test_m
    if i % 10 == 0 :
        train_accs.append(1-numpy.sum(predict(train_xn)==train_yn)*1.0/train_m)
        test_accs.append(1-numpy.sum(predict(test_xn)==test_yn)*1.0/test_m)
        
#-----------------
#plot the learning
#-----------------
if True:
    p1 = pylab.plot(numpy.array(train_accs), label='Train')
    p2 = pylab.plot(numpy.array(test_accs), label='Test')
    pylab.xlabel('Iteration')
    pylab.ylabel('Error')
    pylab.legend()
    pylab.show()