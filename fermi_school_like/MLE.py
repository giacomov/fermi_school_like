import numpy as np
import scipy.optimize
from data_generative_process import DataGenerativeProcess

class Likelihood(object):

    def __init__(self, bins, data, model, noise_model):

        self._bins = bins
        self._data = data
        self._model = model
        self._original_parameters = self._model.current_parameters
        self._noise_model = None
        self._set_noise_model(noise_model)

        assert np.all( (self._data >= 0) ), "Data must be >= 0"

    def _set_noise_model(self, noise_model):

        if noise_model.lower() == 'poisson':

            self._noise_model = 'poisson'

        elif noise_model.lower() == 'gaussian':

            self._noise_model = 'gaussian'

        else:

            raise ValueError("Noise model %s not known. Use 'poisson' or 'gaussian'." % noise_model)

    def _get_noise_model(self):

        return self._noise_model

    noise_model = property(_get_noise_model, _set_noise_model, doc='''Sets or gets the noise model''')

    def profile(self, grid):

        log_like = []

        if self.noise_model == 'poisson':

            likelihood = self._poisson_like

        else:

            likelihood = self._gaussian_like

        for parameter in grid:

            log_like.append(likelihood([parameter]))

        return np.array(log_like)

    def _poisson_like(self, parameters):

        # NB: we drop the -log(n!) part

        prediction = self._model.integral(self._bins.starts, self._bins.stops, parameters)

        idx = prediction > 0

        log_like = np.zeros_like(prediction)

        log_like[idx] = self._data[idx] * np.log(prediction[idx]) - prediction[idx]

        # nidx will contain the opposite of idx, i.e., the indexes of all the bins where
        # prediction == 0 (prediction cannot be negative)

        nidx = ~idx

        if np.any(nidx & (self._data > 0)):

            # We have a prediction of 0 and data !=0, which strictly speaking
            # is impossible. This can happen sometimes during the fit if the fitting engine
            # explore weird parts of the parameter space
            # Just return a very low likelihood so this will not be the maximum of the likelihood
            return -1e9

        else:

            # Here both data and prediction are 0. 0 log(0) = 0, so the likelihood reduces to:
            log_like[nidx] = 0

        return np.sum(log_like)

    def _gaussian_like(self, parameters):

        prediction = self._model.integral(self._bins.starts, self._bins.stops, parameters)

        idx = self._data > 0

        # Given that a variance of 0 is unphysical, when self._data is zero we compute the variance
        # from the model ("model variance")

        variances = np.where(self._data > 0, np.sqrt(self._data), np.sqrt(prediction))

        chisq = np.sum( (self._data - prediction)**2 / variances**2 )

        return -0.5 * chisq

    def maximize(self):

        # Scipy minimizes, does not maximizes
        if self._noise_model == 'poisson':

            minus_log_like = lambda x: - self._poisson_like(x)

        else:

            minus_log_like = lambda x: - self._gaussian_like(x)

        # Initial value: use the current value for the parameters
        initial_values = self._model.current_parameters

        bounds = self._model.bounds

        res = scipy.optimize.minimize(minus_log_like, initial_values, bounds=bounds)

        return np.squeeze(res.x)

    def get_errors(self, best_fit_parameters_, delta = 0.5):
        """
        Get error for the parameter (only one parameter supported).

        :param best_fit_parameters: list of best fit parameters (currently a list of one element)
        :param delta: the delta log like to use as error measurement. Use 0.5 for 1-sigma error
        :return: the positive and negative error
        """

        # Transform in numpy array
        best_fit_parameters = np.array(best_fit_parameters_, ndmin=1)

        # Find the place where the log_likelihood rescaled to zero meets -0.5
        if self._noise_model == 'poisson':

            log_like = self._poisson_like

        else:

            log_like = self._gaussian_like

        # Get the likelihood value at the maximum
        max_log_like = log_like(best_fit_parameters[0])

        # TODO : this only supports a function with one parameter at the moment. Extending this
        # would mean that for each iteration there should be a fit where all other parameters
        # are fitted

        assert len(best_fit_parameters) == 1, "Don't support more than one parameter yet"

        (min_bound, max_bound), = self._model.bounds

        biased_like = lambda x: log_like(x) - max_log_like + delta

        negative_error = scipy.optimize.brentq(biased_like, min_bound, best_fit_parameters[0]) - best_fit_parameters[0]
        positive_error = scipy.optimize.brentq(biased_like, best_fit_parameters[0], 2 * best_fit_parameters[0]) - \
                         best_fit_parameters[0]

        return negative_error, positive_error

    def generate_and_fit(self, data_generative_process, n_iter, compute_errors=False):
        """
        Generate data and fit them for the specified number of times

        :param n_iter: number of times
        :return: array of results
        """

        assert len(self._model.current_parameters) == 1, "Only support one parameter at the moment"

        results = np.zeros(n_iter)

        errors = np.zeros((n_iter, 2))

        for i in range(n_iter):

            # Generate new data set

            data = data_generative_process.generate(self._bins)

            # Set back the fit model to the initial parameters
            self._model.current_parameters = self._original_parameters

            # Build a Likelihood object and maximize the likelihood

            this_like = Likelihood(self._bins, data, self._model, self._noise_model)

            this_result = this_like.maximize()

            results[i] = this_result

            if compute_errors:

                this_errors = this_like.get_errors(this_result)

                errors[i] = this_errors


        if compute_errors:

            return results, errors

        else:

            return results