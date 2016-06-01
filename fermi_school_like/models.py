import numpy as np

class Constant(object):

    def __init__(self, a):

        self._a = a

    @property
    def a(self):
        return self._a

    def _get_current_parameters(self):
        return list([self._a])

    def _set_current_parameters(self, new_parameters):
        self._a = new_parameters

    current_parameters = property(_get_current_parameters, _set_current_parameters,
                                  doc='Sets or gets current parameters')

    @property
    def bounds(self):
        return ((1e-3, None),)

    def __call__(self, x):

        # The 1e-20 is to avoid the results to be exactly 0, which would cause problems
        # in the log-likelihood call because of the logarithm of M

        results = np.zeros_like(x) + self._a

        return results

    def evaluate(self, x, parameters):
        assert len(parameters) == 1, "Wrong number of parameters"

        a = parameters

        self._a = a

        return self.__call__(x)

    def integral(self, lower_bounds, upper_bounds, parameters=None):

        if parameters:

            self._a = parameters

        return self._a * (upper_bounds - lower_bounds)