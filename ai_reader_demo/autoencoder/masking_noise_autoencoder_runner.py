"""
DOCSTRING
"""
import autoencoder.autoencoder_models.denoising_autoencoder
import numpy
import sklearn.preprocessing
import tensorflow

mnist = tensorflow.examples.tutorials.mnist.input_data.read_data_sets(
    'MNIST_data', one_hot = True)

def get_random_block_from_data(data, batch_size):
    """
    DOCSTRING
    """
    start_index = numpy.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

def standard_scale(X_train, X_test):
    """
    DOCSTRING
    """
    preprocessor = sklearn.preprocessing.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
n_samples = int(mnist.train.num_examples)
training_epochs = 100
batch_size = 128
display_step = 1

autoencoder = autoencoder.autoencoder_models.denoising_autoencoder.MaskingNoiseAutoencoder(
    n_input = 784,
    n_hidden = 200,
    transfer_function = tensorflow.nn.softplus,
    optimizer = tensorflow.train.AdamOptimizer(learning_rate = 0.001),
    dropout_probability = 0.95)

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
