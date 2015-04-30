import numpy
import theano
import theano.tensor as T

prob = numpy.array([.05,.05,.95,.5,.5,.95,.95,.95,.05],dtype='f')

theano_rng = T.shared_randomstreams.RandomStreams(numpy.random.randint(2 ** 30))

h = T.fvector("h")
h_sample = theano_rng.binomial(size=(9,), n=1, p=h)

sample = theano.function(inputs=[h,],outputs=[h_sample,])

print sample(prob)