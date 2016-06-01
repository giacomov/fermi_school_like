import numpy as np
from bins import Bins

class DataGenerativeProcess(object):
    """
    A process able to generate data.

    Usage:

    >>> g = DataGenerativeProcess(function)

    where function describes the differential flux of photons as function of x.

    """
    def __init__(self, function):

        self._function = function

    def generate(self, bins):
        """
        Generate data in the bins defined by xs

        :param bins : a Bins instance
        :return: a vector of data
        """

        assert isinstance(bins, Bins)

        predictions = self._function.integral(bins.starts, bins.stops)

        data = np.random.poisson(predictions)

        return data
