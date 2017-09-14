"""This file contains the TimeSeries class.

"""

from pandas import Series

class Timeseries(Series):
    def __init__(self, series, polder, units="m/d"):
        Series.__init__(self)
        self.series = series
        self.polder = polder
        self.units = units

    def flux(self):
        """

        Returns
        -------
        flux: pandas.Series
            Flux voor the series

        """
        flux = self.series.multiply(self.polder.area)
        return flux
