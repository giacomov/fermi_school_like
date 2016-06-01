import numpy as np


class Bins(object):

    def __init__(self, bin_boundaries):

        self._bin_boundaries = np.array(bin_boundaries)

        self._bin_starts = self._bin_boundaries[:-1]
        self._bin_stops = self._bin_boundaries[1:]
        self._bin_centers = (self._bin_stops + self._bin_starts) / 2.0
        self._bin_widths = (self._bin_stops - self._bin_starts)

    @property
    def boundaries(self):

        return self._bin_boundaries

    @property
    def starts(self):

        return self._bin_starts

    @property
    def stops(self):

        return self._bin_stops

    @property
    def centers(self):

        return self._bin_centers

    @property
    def widths(self):

        return self._bin_widths