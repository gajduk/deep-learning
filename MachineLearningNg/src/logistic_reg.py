import theano
import numpy
import pylab
import theano.tensor as T

#--------------------
#set some parameters
#--------------------
#learning rate
alpha = .01
max_iterations = 15000
#--------------------
#load the data
#--------------------
from numpy import genfromtxt
my_data = genfromtxt('ex2data1.txt', delimiter=',')
m,n = my_data.shape
#features and labels
xn = my_data[:,0:n-1]
yn = my_data[:,n-1]

#visualize the data
if True:
    pylab.scatter(my_data[yn==0,0],my_data[yn==0,1],marker='+')
    pylab.scatter(my_data[yn==1,0],my_data[yn==1,1])
    #pylab.show()
    
#normalize the data
mux = numpy.mean(xn, axis=0)
sigmax = numpy.std(xn,axis=0)
xn = numpy.divide(xn-mux,sigmax)

#--------------------
#declare theano variables
#--------------------
X = theano.tensor.matrix('X', dtype='floatX')
Y = theano.tensor.matrix('Y', dtype='floatX')
theta = theano.shared(numpy.zeros((n,1)), name="theta")

#--------------------
#declare theano expressions for logistic regression regression
#--------------------
#hypothesis
h = T.nnet.sigmoid(T.dot(X,theta))
#cost function
cost = 1.0/m*T.sum(-Y*T.log(h)-(1-Y)*T.log(1-h))
#grad function
gtheta = T.grad(cost,theta)
#train function
train = theano.function(inputs=[X,Y],outputs=[cost,],updates=((theta, theta-alpha*gtheta),))

#predict function
predict = theano.function(inputs=[X,],outputs=[h>0.5,])
#-----------------
#train the logistic regression
#-----------------
#add bias features
input_x = numpy.ones((m,n))
input_x[:,1:n] = xn

yn = yn.reshape(m,1)
costs = []
gradient = theano.function(inputs=[X,Y],outputs=[gtheta,],on_unused_input='warn')

for i in range(max_iterations):
    cost = train(input_x,yn)  
    costs.append(cost)
   
#-----------------
#plot the learning
#-----------------
if False:
    pylab.plot(numpy.array(costs))
    pylab.show()
    
if True:
    t0 = theta.get_value()[0]
    t1 = theta.get_value()[1]
    t2 = theta.get_value()[2]
    print t0,t1,t2
    x_plot = numpy.arange(-3,3,0.01)
    y_plot = 1.0/t2*(-x_plot*t2-t0)
    pylab.plot(x_plot*sigmax[0]+mux[0],y_plot*sigmax[1]+mux[1])
    pylab.show()