"""
DOCSTRING
"""
import autoencoder.utils
import numpy
import tensorflow

class AdditiveGaussianNoiseAutoencoder:
    """
    DOCSTRING
    """
    def __init__(
        self,
        n_input,
        n_hidden,
        transfer_function=tensorflow.nn.softplus,
        optimizer=tensorflow.train.AdamOptimizer(),
        scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tensorflow.placeholder(tensorflow.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        # model
        self.x = tensorflow.placeholder(tensorflow.float32, [None, self.n_input])
        self.hidden = self.transfer(tensorflow.add(tensorflow.matmul(
            self.x + scale * tensorflow.random_normal((n_input,)),
            self.weights['w1']), self.weights['b1']))
        self.reconstruction = tensorflow.add(tensorflow.matmul(
            self.hidden, self.weights['w2']), self.weights['b2'])
        # cost
        self.cost = 0.5 * tensorflow.reduce_sum(tensorflow.pow(
            tensorflow.sub(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        init = tensorflow.initialize_all_variables()
        self.sess = tensorflow.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        """
        DOCSTRING
        """
        all_weights = dict()
        all_weights['w1'] = tensorflow.Variable(
            autoencoder.utils.xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tensorflow.Variable(
            tensorflow.zeros([self.n_hidden], dtype=tensorflow.float32))
        all_weights['w2'] = tensorflow.Variable(
            tensorflow.zeros([self.n_hidden, self.n_input], dtype=tensorflow.float32))
        all_weights['b2'] = tensorflow.Variable(
            tensorflow.zeros([self.n_input], dtype=tensorflow.float32))
        return all_weights

    def calc_total_cost(self, X):
        """
        DOCSTRING
        """
        return self.sess.run(self.cost, feed_dict = \
            {self.x: X, self.scale: self.training_scale})

    def generate(self, hidden = None):
        """
        DOCSTRING
        """
        if hidden is None:
            hidden = numpy.random.normal(size = self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})

    def get_biases(self):
        """
        DOCSTRING
        """
        return self.sess.run(self.weights['b1'])

    def get_weights(self):
        """
        DOCSTRING
        """
        return self.sess.run(self.weights['w1'])

    def partial_fit(self, X):
        """
        DOCSTRING
        """
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = \
            {self.x: X, self.scale: self.training_scale})
        return cost

    def reconstruct(self, X):
        """
        DOCSTRING
        """
        return self.sess.run(self.reconstruction, feed_dict = \
            {self.x: X, self.scale: self.training_scale})

    def transform(self, X):
        """
        DOCSTRING
        """
        return self.sess.run(self.hidden, feed_dict = \
            {self.x: X, self.scale: self.training_scale})

class MaskingNoiseAutoencoder:
    """
    DOCSTRING
    """
    def __init__(
        self,
        n_input,
        n_hidden,
        transfer_function=tensorflow.nn.softplus,
        optimizer=tensorflow.train.AdamOptimizer(),
        dropout_probability=0.95):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.dropout_probability = dropout_probability
        self.keep_prob = tensorflow.placeholder(tensorflow.float32)
        network_weights = self._initialize_weights()
        self.weights = network_weights
        # model
        self.x = tensorflow.placeholder(tensorflow.float32, [None, self.n_input])
        self.hidden = self.transfer(tensorflow.add(tensorflow.matmul(
            tensorflow.nn.dropout(self.x, self.keep_prob),
            self.weights['w1']), self.weights['b1']))
        self.reconstruction = tensorflow.add(tensorflow.matmul(
            self.hidden, self.weights['w2']), self.weights['b2'])
        # cost
        self.cost = 0.5 * tensorflow.reduce_sum(tensorflow.pow(
            tensorflow.sub(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        init = tensorflow.initialize_all_variables()
        self.sess = tensorflow.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        """
        DOCSTRING
        """
        all_weights = dict()
        all_weights['w1'] = tensorflow.Variable(
            autoencoder.utils.xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tensorflow.Variable(
            tensorflow.zeros([self.n_hidden], dtype=tensorflow.float32))
        all_weights['w2'] = tensorflow.Variable(
            tensorflow.zeros([self.n_hidden, self.n_input], dtype=tensorflow.float32))
        all_weights['b2'] = tensorflow.Variable(
            tensorflow.zeros([self.n_input], dtype=tensorflow.float32))
        return all_weights

    def calc_total_cost(self, X):
        """
        DOCSTRING
        """
        return self.sess.run(self.cost, feed_dict={self.x: X, self.keep_prob: 1.0})

    def generate(self, hidden=None):
        """
        DOCSTRING
        """
        if hidden is None:
            hidden = numpy.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def get_biases(self):
        """
        DOCSTRING
        """
        return self.sess.run(self.weights['b1'])

    def get_weights(self):
        """
        DOCSTRING
        """
        return self.sess.run(self.weights['w1'])

    def partial_fit(self, X):
        """
        DOCSTRING
        """
        cost, opt = self.sess.run(
            (self.cost, self.optimizer),
            feed_dict={self.x: X, self.keep_prob: self.dropout_probability})
        return cost

    def reconstruct(self, X):
        """
        DOCSTRING
        """
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.keep_prob: 1.0})

    def transform(self, X):
        """
        DOCSTRING
        """
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.keep_prob: 1.0})
