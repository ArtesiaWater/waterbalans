"""This file contains the different classes for the buckets.



"""

from abc import ABC
from collections import OrderedDict

import pandas as pd

from waterbalans.io import load_series


class Bucket:
    __doc__ = """Class to construct a Bucket instance from a string. 

    """

    def __new__(cls, kind=None, *args, **kwargs):
        return eval(kind)(*args, **kwargs)


class BucketBase(ABC):
    __doc__ = """Base class from which all bucket classes inherit.
    
    """

    def __init__(self, polder=None, data=None):
        self.polder = polder  # Reference to mother object.
        self.data = data
        self.series = OrderedDict()
        # self.load_series()

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


class verhard(BucketBase):
    def __init__(self, polder, data):
        BucketBase.__init__(self, polder, data)


class onverhard(BucketBase):
    def __init__(self, polder, data):
        BucketBase.__init__(self, polder, data)


class water(BucketBase):
    def __init__(self, polder, data):
        BucketBase.__init__(self, polder, data)
