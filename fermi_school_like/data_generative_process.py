import numpy as np

class DataGenerativeProcess(object):
    """
    A process able to generate data.

    Usage:

    >>> g = DataGenerativeProcess(function)

    where function describes the differential flux of photons as function of x.

    """
    def __init__(self, function):

        self._function = function

    def generate(self, xs):
        """
        Generate n_bins of data between minimum and maximum

        :param minimum: minimum for the generation
        :param maximum: maximum for the generation
        :param n_bins: number of bins to generate
        :return: a vector of data
        """

        predictions = self._function(xs)

        data = np.random.poisson(predictions)

        return data
