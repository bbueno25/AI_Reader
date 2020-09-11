"""
DOCSTRING
"""
import numpy
import tensorflow

def xavier_init(fan_in, fan_out, constant = 1):
    """
    DOCSTRING
    """
    low = -constant * numpy.sqrt(6.0 / (fan_in + fan_out))
    high = constant * numpy.sqrt(6.0 / (fan_in + fan_out))
    return tensorflow.random_uniform(
        (fan_in, fan_out), minval=low, maxval=high, dtype=tensorflow.float32)
