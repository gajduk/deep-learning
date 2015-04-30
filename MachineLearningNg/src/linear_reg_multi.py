import theano
import numpy

#--------------------
#set some parameters
#--------------------
#learning rate
alpha = .01
max_iterations = 1500
#--------------------
#load the data
#--------------------
from numpy import genfromtxt
my_data = genfromtxt('ex1data2.txt', delimiter=',')
#features and labels
xn = my_data[:,0:2]
yn = my_data[:,2]
#number fo training examples
m,n = my_data.shape
#normalize the data
mux = numpy.mean(xn, axis=0)
sigmax = numpy.std(xn,axis=0)
xn = numpy.divide(xn-mux,sigmax)

#--------------------
#declare theano variables
#--------------------
X = theano.tensor.dmatrix('X')
Y = theano.tensor.dmatrix('Y')
theta = theano.shared(numpy.zeros((n,1)), name="theta")

#--------------------
#declare theano expressions for linear regression
#--------------------
#hypothesis
h = theano.tensor.dot(X,theta)
#cost function
temp = 1.0/2/m
cost = temp*theano.tensor.sum(theano.tensor.sqr(h-Y))
#grad function
gtheta = theano.tensor.grad(cost,theta)
#train function
train = theano.function(inputs=[X,Y],outputs=[cost,],updates=((theta, theta-alpha*gtheta),))

#predict function
predict = theano.function(inputs=[X,],outputs=[h,])
#-----------------
#train the linear regression
#-----------------
#add bias features
input_x = numpy.ones((m,n))
input_x[:,1:n] = xn
print input_x.shape
yn = yn.reshape(m,1)
costs = []
for i in range(max_iterations):
   cost = train(input_x,yn)     
   costs.append(cost)
   
#-----------------
#plot the learning
#-----------------
import pylab
pylab.plot(numpy.array(costs))
pylab.show()