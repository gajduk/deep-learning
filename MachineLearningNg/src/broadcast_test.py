import theano
import numpy
import theano.tensor as T

bn = numpy.array([5,6,100,200])
bn = bn.reshape(2,2)
print "Bn shape",bn.shape
x = T.matrix('x')
b = theano.shared(bn)
y = T.outer(x,b)
f = theano.function(inputs=[x],outputs=[y,])
xn = numpy.array([1,2,3,4])
xn = xn.reshape(2,2)
print "Xn shape",xn.shape
print f(xn)