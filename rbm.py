import numpy
import math
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import time


class RBM:
    def __init__(
            self,
            num_visible,
            num_hidden,
            learning_rate=0.1
    ):
        # Initial state of each variable
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.learning_rate = learning_rate

        # Create Random generator
        self.numpy_rng = numpy.random.RandomState(1234)
        self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))

        # Initial Weight which is uniformely sampled from -4*sqrt(6./(n_visible+n_hidden)) and 4*sqrt(6./(n_hidden+n_visible))
        tmpWeight = numpy.asarray(
            self.numpy_rng.uniform(
                low=-4 * numpy.sqrt(6. / (self.num_visible + self.num_hidden)),
                high=4 * numpy.sqrt(6. / (self.num_visible + self.num_hidden)),
                size=(self.num_visible, self.num_hidden)
            ),
            dtype=theano.config.floatX
        )
        # theano shared variables for weights
        self.weights = theano.shared(value=tmpWeight, name='weight', borrow=True)

        # Inital theano shared variables hidden Bias
        self.hbias = theano.shared(
            value=numpy.zeros(
                self.num_hidden,
                dtype=theano.config.floatX
            ),
            name='hbias',
            borrow=True
        )

        # Inital theano shared variables visible Bias
        self.vbias = theano.shared(
            value=numpy.zeros(
                self.num_visible,
                dtype=theano.config.floatX
            ),
            name='vbias',
            borrow=True
        )

    # Calculate and return Positive hidden states and probs
    def positiveProb(self, visible):
        pos_hidden_activations = T.dot(visible, self.weights) + self.hbias
        pos_hidden_probs = T.nnet.sigmoid(pos_hidden_activations)
        pos_hidden_states = self.theano_rng.binomial(
            size=pos_hidden_probs.shape,
            n=1,
            p=pos_hidden_probs,
            dtype=theano.config.floatX
        )
        return [pos_hidden_states, pos_hidden_probs]

    # Calculate and return Negative hidden states and probs
    def negativeProb(self, data, hidden, k=1):
        for i in range(k):
            v1_activations = T.dot(hidden, self.weights.T) + self.vbias
            v1_probs = T.nnet.sigmoid(v1_activations)
            v1_sample = self.theano_rng.binomial(
                size=v1_probs.shape,
                n=1,
                p=v1_probs,
                dtype=theano.config.floatX
            )
            # # To ignore the missing visible unit
            # v1_probs = data * v1_probs
            # Get back to calculate hidden again
            hidden, hidden_probs = self.positiveProb(v1_probs)
        return [v1_probs, hidden_probs,hidden]

    # Get hidden state
    def getHidden(self, visible):
        hidden_activations = T.dot(visible, self.weights) + self.hbias
        hidden_probs = T.nnet.sigmoid(hidden_activations)
        hidden_states = self.theano_rng.binomial(
            size=hidden_probs.shape,
            n=1,
            p=hidden_probs,
            dtype=theano.config.floatX
        )
        return hidden_states,hidden_probs

    # Get visivle state
    def getVisible(self, hidden):
        visible_activations = T.dot(hidden, self.weights.T) + self.vbias
        visible_probs = T.nnet.sigmoid(visible_activations)
        visible_states = self.theano_rng.binomial(
            size=visible_probs.shape,
            n=1,
            p=visible_probs,
            dtype=theano.config.floatX
        )
        return visible_probs

    # Train RMB model
    def train(self, data, max_epochs=1000, batch_size=10, step=1):
        for epoch in range(max_epochs):

            # Divide in to minibatch
            total_batch = int(math.ceil(data.shape[0] / batch_size))

            # Loop for each batch
            for batch_index in range(total_batch):
                # Get the data for each batch
                tmpData = data[batch_index * batch_size: (batch_index + 1) * batch_size]
                num_examples = tmpData.shape[0]

                # Caculate positive probs and Expectation for Sigma(ViHj) data
                pos_hidden_states, pos_hidden_probs = self.positiveProb(tmpData)
                pos_associations = T.dot(tmpData.T, pos_hidden_probs)

                # Calculate negative probs and Expecatation for Sigma(ViHj) recon with k = 1,....
                neg_visible_probs, neg_hidden_probs,neg_hidden_states = self.negativeProb(tmpData, pos_hidden_states, k=step)
                neg_associations = T.dot(neg_visible_probs.T, neg_hidden_probs)

                # Update weight
                self.weights += self.learning_rate * ((pos_associations - neg_associations) / num_examples)

                self.vbias += self.learning_rate * 0.4 * (T.sum(tmpData.T - neg_visible_probs.T,axis=1)/ num_examples)
                self.hbias += self.learning_rate * 0.4 *(T.sum(pos_hidden_states.T - neg_hidden_states.T,axis=1)/ num_examples)


            print('Epoch: {0}'.format(epoch))
            #
            # # Check the error only for the one that not missing
            # # error = math.sqrt(numpy.sum((data - tmpVisible.eval()) ** 2) / numpy.sum(data == 1))
            # error=math.sqrt(numpy.sum((data-tmpVisible.eval())**2)/numpy.size(data))
            # print ('Epoch : {0}, RMSE : {1}'.format(epoch, 0))

        # print (neg_hidden_states.eval())
        # print (pos_associations.eval())
        # print (pos_hidden_probs.eval())
        # print (pos_hidden_states.eval())
    #calculate distance between papers
    def recommend(self,testcase,data,w,hb,Rank=5,):
        testHidden = self.getHiddenPro(testcase,w,hb)
        tmpHidden = self.getHiddenPro(data,w,hb)
        gap = (tmpHidden - testHidden) ** 2
        distance=numpy.sqrt(numpy.sum(gap,axis=1))
        ind=numpy.argsort(distance)[:Rank]
        return ind

    def getHiddenPro(self,visible,w,hb):
        hidden_activations = numpy.dot(visible, w) + hb
        hidden_probs = 1.0/(1+numpy.exp(-hidden_activations))
        return hidden_probs

if __name__ == '__main__':
    # print(rbm.weights.get_value())

    data=numpy.load(open("data-full.bin","rb"))
    test_data=data["input_data"]#first 50 data size
    train_data=data["train_data"]# total 16980 data size

    second_part_data=train_data[:50]
    train_data=train_data[50:1050]
    print ("train data set number: {}".format(len(train_data)))

    vocabulary_num=len(train_data[0])

    topic_num=150
    rbm = RBM(num_visible=vocabulary_num, num_hidden=topic_num)
    rbm.train(train_data, max_epochs=8, batch_size=80,step=1)
    start = time.clock()
    w=rbm.weights.eval()
    hb=rbm.hbias.eval()
    print (w.shape)
    print (hb.shape)
    end = time.clock()
    print ("time: %f s" % (end - start))
    start = time.clock()
    c10=0
    c5=0
    for i in range(50):
        ind=rbm.recommend(test_data[i],second_part_data,w,hb,Rank=10)
        if i in ind:
            c10+=1
        if i in ind[:5]:
            c5+=1
        print ("list: {0},index: {1}".format(ind,i))
    print ("Top10 accuracy: {0},percent: {1}".format(c10, c10 * 1.0 / 50))
    print ("Top 5 accuracy: {0},percent: {1}".format(c5, c5 * 1.0 / 50))
    end = time.clock()
    print ("time: %f s" % (end - start))

    # rbm.train(data, max_epochs=10, step=3)
    # rbm.train(data, max_epochs=10, step=5)
    # rbm.train(data, max_epochs=10, step=9)
    print ("finish!")