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
my_data = genfromtxt('ex1data1.txt', delimiter=',')
#features and labels
xn = my_data[:,0]
yn = my_data[:,1]
#number fo training examples
m,n = my_data.shape



#--------------------
#declare theano variables
#--------------------
X = theano.tensor.dmatrix('X')
Y = theano.tensor.dmatrix('Y')
theta = theano.shared(numpy.zeros((2,1)), name="theta")

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
input_x = numpy.ones((m,2))
input_x[:,1] = xn
yn = yn.reshape(m,1)
costs = []
for i in range(max_iterations):
   cost = train(input_x,yn)     
   costs.append(cost)


#--------------------
#plot the data
#--------------------
import pylab
pylab.scatter(xn,yn)
x_plot = numpy.arange(0,25,.1)
m1 = x_plot.shape[0]

x_qwe = numpy.ones((m1,2))
x_qwe[:,1] = x_plot
predictions, = predict(x_qwe)
pylab.plot(x_plot,predictions)
pylab.ylabel('Profit in $10,000s')
pylab.xlabel('Population of City in 10,000s')
pylab.show()