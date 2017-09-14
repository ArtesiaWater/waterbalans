"""This file contains the polder class

"""

from collections import OrderedDict

import pandas as pd

from .io.base import load_model
from .subpolder import SubPolder
from .timeseries import Timeseries


class Polder:
    __doc__ = """The Polder class is the main object for a waterbalance.
    
    """

    def __init__(self, id, db_model="shp", db_series="xml", db_param="shp"):
        self.id = id

        # Store the database source types
        self.db_model = db_model  # model database
        self.db_series = db_series
        self.db_param = db_param

        # 1. Create the model structure
        self.subpolders = OrderedDict()
        self.load_model()

        # 2. Load all series
        self.series = OrderedDict()
        self.load_series()

        # 3. Load all parameters
        self.parameters = pd.DataFrame()
        self.load_parameters()

    def load_model(self):
        """Method to import model structure description from a file or
        database.

        """
        data = load_model(fname=None, id=self.id, db_model=self.db_model)

        self.name = None
        self.area = 0.0
        self.x = 0.0
        self.y = 0.0

        for id in data["eag"]:
            subpolder = SubPolder(id=id, polder=self)
            self.subpolders[id] = subpolder

    def load_series(self):
        """Method to import the time series in both the Polder object and
        the subpolder Objects.

        """
        self.series["prec"] = Timeseries(pd.Series(), self)
        self.series["evap"] = Timeseries(pd.Series(), self)

        for subpolder in self.subpolders.values():
            subpolder.load_series()

    def load_parameters(self):
        pass

    def calculate_wb(self):
        """Method to calculate the waterbalance for the Polder.

        Returns
        -------

        """
        pass

    def validate_wb(self):
        """Method to validate the water balance based on the total input,
        output and the change in storage of the model for each time step.

        Returns
        -------

        """
        pass

    def dump_data(self):
        pass

    def dump(self):
        pass
