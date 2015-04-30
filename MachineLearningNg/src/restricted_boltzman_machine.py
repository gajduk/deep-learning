import numpy
import theano
import theano.tensor as T
import gzip
import cPickle
import matplotlib.pyplot as plt
import math
import pickle

def display_weights(W,width,height,epoch):
    W = W.get_value()
    min_W = numpy.min(W)
    max_W = numpy.max(W)
    W = (W-min_W)/(max_W-min_W)
    h,n = W.shape
    per_row = int(math.floor(math.sqrt(h)))
    rows = int(math.ceil(h*1.0/per_row))
    border = 3
    res = numpy.zeros((rows*(height+border),per_row*(width+border),3))
    for r in range(rows):
        for c in range(per_row):
            i = per_row*r+c
            if i < h:
                single_neuron = W[i,:]
                res[r*(height+border):(r+1)*(height+border)-border,c*(width+border):(c+1)*(width+border)-border,:] = numpy.swapaxes(numpy.tile(single_neuron.reshape(width,height),[3,1,1]),0,2)
    plt.imshow(res)
    plt.savefig("epoch_"+str(epoch)+".png")

def loadData():
    f = gzip.open("mnist.pkl.gz", 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return (train_set, valid_set, test_set)

def train():
    train_set, valid_set, test_set = loadData()
    x,y = train_set
    m,n_input = x.shape
    width = 28
    height = 28
    n_hidden = 49
    
    learning_rate = .1
    
    #set up shared variables
    W = theano.shared(numpy.random.uniform(-4 * numpy.sqrt(6. / (n_hidden + n_input)),4 * numpy.sqrt(6. / (n_hidden + n_input)),(n_hidden,width*height)),name="W")
    b_v = theano.shared(numpy.zeros((width*height,)),name="b_v")
    b_h = theano.shared(numpy.zeros((n_hidden,)),name="b_h")
    
    theano_rng = T.shared_randomstreams.RandomStreams(numpy.random.randint(2 ** 30))
    
    v_input = T.fvector("v_input")
    
    #1. sample hidden units
    h_prob = T.nnet.sigmoid(T.dot(v_input,W.T)+b_h)
    h_sample = theano_rng.binomial(size=(n_hidden,), n=1, p=h_prob)
    #2. calculate positive gradient
    g_p = T.outer(v_input,h_sample)
    #3. make reconstruction
    v_prob_reconstruction = T.nnet.sigmoid(T.dot(h_sample,W)+b_v)
    v_reconstruction = theano_rng.binomial(size=(n_input,), n=1, p=v_prob_reconstruction)
    h_prob_reconstruction = T.nnet.sigmoid(T.dot(v_reconstruction,W.T)+b_h)
    h_reconstruction = theano_rng.binomial(size=(n_hidden,), n=1, p=h_prob_reconstruction)
    #4. calculate negative gradient
    g_n = T.outer(v_reconstruction,h_reconstruction)
    #FUNCTIONS FOR TESTING
    #f_h_prob = theano.function(inputs=[v_input,],outputs=[h_prob,])
    #f_h_sample = theano.function(inputs=[v_input,],outputs=[h_sample,])
    #f_g_p = theano.function(inputs=[v_input,],outputs=[g_p,])
    #f_v_prob_reconstruction = theano.function(inputs=[v_input,],outputs=[v_prob_reconstruction,])
    #f_v_reconstruction = theano.function(inputs=[v_input,],outputs=[v_reconstruction,])
    #f_h_prob_reconstruction = theano.function(inputs=[v_input,],outputs=[h_prob_reconstruction,])
    #f_h_reconstruction = theano.function(inputs=[v_input,],outputs=[h_reconstruction,])
    #f_g_n = theano.function(inputs=[v_input,],outputs=[g_n,])
    
    learn = theano.function(inputs=[v_input,],updates=[(W,W+learning_rate*(g_p-g_n).T)])
    
    for i in range(300001):
        if i > 0:
            if i%10000 == 0:
                print "Epcoh: ",i
                display_weights(W,width,height,i)
        learn(x[i%m,:])
    
    with open('weights.pkl', 'wb') as output:
        pickle.dump(W.get_value(), output, pickle.HIGHEST_PROTOCOL)
        
def extractFeatures():
    n_hidden = 49
    train_set, valid_set, test_set = loadData()
    with open('weights.pkl', 'rb') as input:
        W_values = pickle.load(input)
    v_input = T.fvector("v_input")
    W = theano.shared(W_values,name="W")
    b_h = theano.shared(numpy.zeros((n_hidden,)),name="b_h")
    h_prob = T.nnet.sigmoid(T.dot(v_input,W.T)+b_h)
    f_h_prob = theano.function(inputs=[v_input,],outputs=[h_prob,])
    x,y = test_set
    n,m = x.shape
    cool_features = numpy.zeros((n,n_hidden+1))
    for i in range(n):
        cool_features[i,:-1] = f_h_prob(x[i,:])[0]
    cool_features[:,-1] = y
    with open("test_minst_real.arff",'a') as output:
        output.write("@relation contact-lenses")
        output.write("\n\n")
        for i in range(n_hidden):
            output.write("@attribute hidden_feature_"+str(i)+"\n")
        output.write("@attribute class {0,1,2,3,4,5,6,7,8,9}\n")
        output.write("\n\n@data\n")
        numpy.savetxt(output,cool_features, delimiter=",")

def extract_features_simple():
    n_hidden = 49
    train_set, valid_set, test_set = loadData()
    with open('weights.pkl', 'rb') as input:
        W_values = pickle.load(input)
    v_input = T.fvector("v_input")
    W = theano.shared(W_values,name="W")
    b_h = theano.shared(numpy.zeros((n_hidden,)),name="b_h")
    h_prob = T.nnet.sigmoid(T.dot(v_input,W.T)+b_h)
    f_h_prob = theano.function(inputs=[v_input,],outputs=[h_prob,])
    x,y = test_set
    n,m = x.shape
    qwe = numpy.zeros((n,m+1))
    qwe[:,:-1] = x
    qwe[:,-1] = y
    with open("test_simple.arff",'a') as output:
        output.write("@relation minst")
        output.write("\n\n")
        for i in range(m):
            output.write("@attribute pixel_"+str(i)+" numeric\n")
        output.write("@attribute class {0,1,2,3,4,5,6,7,8,9}\n")
        output.write("\n\n@data\n")
        numpy.savetxt(output,qwe,delimiter=",")
    
extractFeatures()
    