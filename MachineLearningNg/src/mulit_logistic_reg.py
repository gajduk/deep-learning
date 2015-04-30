import theano
import numpy
import pylab
import theano.tensor as T
rng = numpy.random

#--------------------
#set some parameters
#--------------------
#learning rate
alpha = 1
max_iterations = 10000
#number of classes
k = 10
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
input_x = numpy.ones((m,n))
input_x[:,1:n] = xn

test_idxs = rng.randint(0,m,1110)
test_xn = input_x[test_idxs,:]
test_yn = yn[test_idxs].astype(int)
numpy.array(range(0,m))
train_idxs = numpy.setdiff1d(numpy.arange(0,m,1),test_idxs).astype(int)
train_xn = input_x[train_idxs,:]
train_yn = yn[train_idxs].astype(int)
train_m = train_xn.shape[0]
test_m = test_xn.shape[0]

#--------------------
#declare theano variables
#--------------------
X = theano.tensor.matrix('X', dtype='floatX')
Y = theano.tensor.matrix('Y', dtype='int32')
#theta = theano.shared(numpy.zeros((n,k)), name="theta",borrow=True)
theta = theano.shared(rng.randn(n,k), name="theta",borrow=True)

#--------------------
#declare theano expressions for logistic regression regression
#--------------------
#hypothesis
p_y_x = T.nnet.softmax(T.dot(X,theta))
predict_y = T.argmax(p_y_x,axis=1)
#cost function from tutorial
#cost = -T.mean(T.log(p_y_x[T.arange(m),Y]))
#cost function from Ng
cost = 1.0/train_m*T.sum(-Y*T.log(p_y_x)-(1-Y)*(T.log(1-p_y_x)))
#grad function
gtheta = T.grad(cost,theta)
#train function
train = theano.function(inputs=[X,Y],outputs=[cost,],updates=((theta, theta-alpha*gtheta),))

gradient = theano.function(inputs=[X,Y],outputs=[gtheta,],on_unused_input='warn')

#predict function
predict = theano.function(inputs=[X,],outputs=[predict_y,])
#probability distribution function
p_y_x_func = theano.function(inputs=[X,],outputs=[p_y_x,])

#-----------------
#train the logistic regression
#-----------------
yn = yn.astype(int)
#cast yn for Ng type learning
input_y = numpy.zeros((train_m,k))
input_y[range(train_m),train_yn] = 1
input_y = input_y.astype(int)
costs = []
train_accs = []
test_accs = []
for i in range(max_iterations):
    if i % 100 == 0:
        print "Iteration:",i
        print "Train accuracy:",numpy.sum(predict(train_xn)==train_yn)*1.0/train_m
        print "Test accuracy:",numpy.sum(predict(test_xn)==test_yn)*1.0/test_m
    if i % 10 == 0 :
        train_accs.append(1-numpy.sum(predict(train_xn)==train_yn)*1.0/train_m)
        test_accs.append(1-numpy.sum(predict(test_xn)==test_yn)*1.0/test_m)
    #print gradient(input_x,input_y)
    cost = train(train_xn,input_y)
    costs.append(cost)
   
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
