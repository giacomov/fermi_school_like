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

    def __call__(self, x):

        results = np.zeros_like(x) + self._a

        return results

    def evaluate(self, x, parameters):
        assert len(parameters) == 2, "Wrong number of parameters"

        a = parameters

        self._a = a

        return self.__call__(x)

class Line(object):

    def __init__(self, a, b):

        self._a = a
        self._b = b

    @property
    def a(self):

        return self._a

    @property
    def b(self):

        return self._b

    def _get_current_parameters(self):

        return list([self._a, self._b])

    def _set_current_parameters(self, new_parameters):

        self._a, self._b = new_parameters

    current_parameters = property(_get_current_parameters, _set_current_parameters,
                                  doc='Sets or gets current parameters')

    def __call__(self, x):

        return np.maximum(self._a * x + self._b, 1e-20)

    def evaluate(self, x, parameters):

        assert len(parameters) == 2, "Wrong number of parameters"

        a, b = parameters

        self._a = a
        self._b = b

        return self.__call__(x)