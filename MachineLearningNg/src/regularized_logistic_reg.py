import theano
import numpy
import pylab
import theano.tensor as T

#--------------------
#set some parameters
#--------------------
#learning rate
alpha = 5
#regularization parameter
l = 2.0
max_iterations = 3500
#--------------------
#load the data
#--------------------
from numpy import genfromtxt
my_data = genfromtxt('ex2data2.txt', delimiter=',')
m,n = my_data.shape
#features and labels
xn = my_data[:,0:n-1]
yn = my_data[:,n-1]

#visualize the data
if True:
    pylab.scatter(my_data[yn==0,0],my_data[yn==0,1],marker='+')
    pylab.scatter(my_data[yn==1,0],my_data[yn==1,1])
    #pylab.show()

#add polynomial features
for degree in range(2,7):
    for c in range(0,degree+1):
        xn = numpy.hstack((xn,(xn[:,0]**(degree-c)*xn[:,1]**c).reshape(m,1)))
        
m,n = xn.shape 
n = n+1
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
cost = 1.0/m*T.sum(-Y*T.log(h)-(1-Y)*T.log(1-h))+l/m/2*T.sum(theta**2)
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
    x = numpy.arange(-1.5,1.5,.005)
    xv,yv = numpy.meshgrid(x,x)
    t1,t2 = xv.shape
    predict_x = numpy.hstack((xv.ravel().reshape(t1*t2,1),yv.ravel().reshape(t1*t2,1)))
    #add polynomial features
    for degree in range(2,7):
        for c in range(0,degree+1):
            predict_x = numpy.hstack((predict_x,(predict_x[:,0]**(degree-c)*predict_x[:,1]**c).reshape(t1*t2,1)))
    predict_x = numpy.divide(predict_x-mux,sigmax)
    predict_x = numpy.hstack((numpy.ones((t1*t2,1)),predict_x))
    zv = numpy.asarray(predict(predict_x)).reshape(t1,t2)
    pylab.contour(xv,yv,zv,1)
    try:
        pylab.show()
    except:
        pass
    