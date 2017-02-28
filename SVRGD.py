"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.W_tilda = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W_tilda',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.b_tilda = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b_tilda',
            borrow=True
        )
        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.p_y_given_x_tilda = T.nnet.softmax(T.dot(input, self.W_tilda) + self.b_tilda)
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        lambda_2 = 0.0001
        L2 = T.sum((self.W)**2)

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) + lambda_2 * L2


    def negative_log_likelihood_tilda(self, y):
        lambda_2 = 0.0001
        L2 = T.sum((self.W_tilda)**2)

        return -T.mean(T.log(self.p_y_given_x_tilda)[T.arange(y.shape[0]), y]) + lambda_2 * L2
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def svrgd_optimization_mnist(learning_rate=0.01, n_epochs=50,
                           dataset='mnist.pkl.gz',
                           batch_size=1):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    batch_size_test = 600
    n_data = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size_test
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size_test

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    #exact_grad_W = numpy.zeros((784, 10),dtype=theano.config.floatX)
    exact_grad_W = T.matrix('exact_grad_W')
    exact_grad_b = T.vector('exact_grad_b')
    #exact_grad_b = numpy.zeros((10,),dtype=theano.config.floatX)
    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)
    cost_tilda = classifier.negative_log_likelihood_tilda(y)

    training_loss = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size_test: (index + 1) * batch_size_test],
            y: train_set_y[index * batch_size_test: (index + 1) * batch_size_test]
        }
    )
    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size_test: (index + 1) * batch_size_test],
            y: test_set_y[index * batch_size_test: (index + 1) * batch_size_test]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size_test: (index + 1) * batch_size_test],
            y: valid_set_y[index * batch_size_test: (index + 1) * batch_size_test]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    g_W_tilda = T.grad(cost=cost_tilda, wrt=classifier.W_tilda)
    g_b_tilda = T.grad(cost=cost_tilda, wrt=classifier.b_tilda)

    calc_exact_grad_W = theano.function(
        inputs = [],
        outputs = g_W_tilda,
        givens = {
            x: train_set_x,
            y: train_set_y
        }
    )
    calc_exact_grad_b = theano.function(
        inputs = [],
        outputs = g_b_tilda,
        givens = {
            x: train_set_x,
            y: train_set_y
        }
    )
    # eval_grad = theano.function(
    #     inputs = [index],
    #     outputs = g_W,
    #     givens = {
    #         x: train_set_x[index],
    #         y: train_set_y[index]
    #     }
    # )

    updates_tilda = [(classifier.W_tilda, classifier.W),
                     (classifier.b_tilda, classifier.b )]
    update_W_tilda = theano.function(
        inputs = [],
        outputs = [],
        updates = updates_tilda
    )


    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * (g_W - g_W_tilda + exact_grad_W)),
               (classifier.b, classifier.b - learning_rate * (g_b - g_b_tilda + exact_grad_b))]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index, exact_grad_W, exact_grad_b],
        outputs=cost,
        updates=updates,

        givens={
            x: train_set_x[index:index+1 ],
            y: train_set_y[index:index+1]
        }

    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    #validation_frequency = min(n_train_batches, patience // 2)
    validation_frequency = n_train_batches
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    #print(n_train_batches)
    #print(validation_frequency)
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    n_epochs = 7
    n_grad = 0
    epoch = 0
    l=-1
    k = 0
    flag = 0
    err_thresh = 0.07

    avg_cost_SVRGD = open("avg_cost_SVRGD.txt", "w")
    gradients = open("gradients.txt", "w")
    while (epoch < n_epochs) :
        epoch = epoch + 1
        #batch_index = numpy.random.choice(n_data-1)
        exact_grad_W_val = calc_exact_grad_W()
        exact_grad_b_val = calc_exact_grad_b()
        n_grad = n_grad + n_data
        # gradients.write(str(n_grad))
        # gradients.write("\n")
        # training_losses = [training_loss(i) for i in range(train_set_x.get_value(borrow=True).shape[0] // batch_size_test)]
        # this_training_loss = numpy.mean(training_losses)
        #
        # avg_cost_SVRGD.write(str(this_training_loss))
        # avg_cost_SVRGD.write("\n")

        print('no of gradients %d' %n_grad)
        # print(sum(exact_grad_W))
        #for minibatch_index in range(n_train_batches):

        l = l + 1
        #for i_in in range(10000*(2**l)):
        for i_in in range(n_data):

            minibatch_index = numpy.random.choice(n_data-1)
            minibatch_avg_cost = train_model(minibatch_index,exact_grad_W_val, exact_grad_b_val )
            n_grad = n_grad + 2
            if (k == n_data-1):
                gradients.write(str(n_grad))
                gradients.write("\n")
                training_losses = [training_loss(i)
                                 for i in range(train_set_x.get_value(borrow=True).shape[0] // batch_size_test)]
                this_training_loss = numpy.mean(training_losses)
                avg_cost_SVRGD.write(str(this_training_loss))
                avg_cost_SVRGD.write("\n")
                # if (this_training_loss < err_thresh):
                #     flag = 1
                #     break

                k = 0

            k = k +1
            # iteration number
            #iter = (epoch - 1) * n_train_batches + minibatch_index
            iter = (epoch-1) * 50000 + i_in
            if (iter + 1) % 50000 == 0:

                # compute zero-one loss on validation set
                print('epoch %i, training loss %f' %(epoch, this_training_loss))
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            # if patience <= iter:
            #     done_looping = True
            #     break
        update_W_tilda()
        if (flag==1):
            break
    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    avg_cost_SVRGD.close()
    gradients.close()
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)


def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = pickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)


if __name__ == '__main__':
    svrgd_optimization_mnist()
