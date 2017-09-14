"""This file contains the different classes for the buckets.

"""

from .io import load_series
from collections import OrderedDict
import pandas as pd


class Bucket:
    __doc__ = """
    
    """
    def __init__(self, polder):
        self.polder = polder  # Reference to mother object.

        self.series = OrderedDict()
        self.load_series()

        self.storage = pd.Series()

        self.parameters = pd.DataFrame(columns=["name", "initial", "optimal"])

        self.area = 0.0  # area in square meters

    def load_series(self):
        series = dict()
        series["prec"] = self.polder.series["prec"]
        series["evap"] = self.polder.series["evap"]

        for name in ["seepage"]:
            series[name] = load_series(name)

        return series

    def calculate_wb(self):
        pass

    def validate_wb(self):
        """Method to validate the water balance based on the total input,
        output and the change in storage of the model for each time step.

        Returns
        -------

        """
        pass


class Verhard(Bucket):
    def __init__(self):
        Bucket.__init__(self)


class Onverhard(Bucket):
    def __init__(self):
        Bucket.__init__(self)


class Water(Bucket):
    def __init__(self):
        Bucket.__init__(self)
