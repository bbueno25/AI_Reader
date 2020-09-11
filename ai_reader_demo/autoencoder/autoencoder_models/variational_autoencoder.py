"""
DOCSTRING
"""
import autoencoder.utils
import numpy
import tensorflow

class VariationalAutoencoder:
    """
    DOCSTRING
    """
    def __init__(self, n_input, n_hidden, optimizer=tensorflow.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        network_weights = self._initialize_weights()
        self.weights = network_weights
        # model
        self.x = tensorflow.placeholder(tensorflow.float32, [None, self.n_input])
        self.z_mean = tensorflow.add(tensorflow.matmul(
            self.x, self.weights['w1']), self.weights['b1'])
        self.z_log_sigma_sq = tensorflow.add(tensorflow.matmul(
            self.x, self.weights['log_sigma_w1']), self.weights['log_sigma_b1'])
        # sample from gaussian distribution
        eps = tensorflow.random_normal(tensorflow.pack(
            [tensorflow.shape(self.x)[0], self.n_hidden]), 0, 1, dtype = tensorflow.float32)
        self.z = tensorflow.add(self.z_mean, tensorflow.mul(
            tensorflow.sqrt(tensorflow.exp(self.z_log_sigma_sq)), eps))
        self.reconstruction = tensorflow.add(tensorflow.matmul(
            self.z, self.weights['w2']), self.weights['b2'])
        # cost
        reconstr_loss = 0.5 * tensorflow.reduce_sum(tensorflow.pow(
            tensorflow.sub(self.reconstruction, self.x), 2.0))
        latent_loss = -0.5 * tensorflow.reduce_sum(
            1 + self.z_log_sigma_sq - tensorflow.square(self.z_mean) \
                - tensorflow.exp(self.z_log_sigma_sq), 1)
        self.cost = tensorflow.reduce_mean(reconstr_loss + latent_loss)
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
        all_weights['log_sigma_w1'] = tensorflow.Variable(
            autoencoder.utils.xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tensorflow.Variable(
            tensorflow.zeros([self.n_hidden], dtype=tensorflow.float32))
        all_weights['log_sigma_b1'] = tensorflow.Variable(
            tensorflow.zeros([self.n_hidden], dtype=tensorflow.float32))
        all_weights['w2'] = tensorflow.Variable(tensorflow.zeros(
            [self.n_hidden, self.n_input], dtype=tensorflow.float32))
        all_weights['b2'] = tensorflow.Variable(tensorflow.zeros(
            [self.n_input], dtype=tensorflow.float32))
        return all_weights

    def calc_total_cost(self, X):
        """
        DOCSTRING
        """
        return self.sess.run(self.cost, feed_dict={self.x: X})

    def generate(self, hidden=None):
        """
        DOCSTRING
        """
        if hidden is None:
            hidden = numpy.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.z_mean: hidden})

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
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def reconstruct(self, X):
        """
        DOCSTRING
        """
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def transform(self, X):
        """
        DOCSTRING
        """
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
