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

    def __init__(self, id=None, db_model="shp", db_series="xml",
                 db_param="shp"):
        self.id = id

        # Store the database source types
        self.db_model = db_model  # model database
        self.db_series = db_series
        self.db_param = db_param

        # Placeholder
        self.data = pd.DataFrame()
        self.subpolders = OrderedDict()
        self.parameters = pd.DataFrame()
        self.series = OrderedDict()

    def _load_model_data(self, fname, id=None):
        """Method to load the data model from a file or database.

        Returns
        -------
        data: pandas.DataFrame
            Imported table that is used as a template to construct the model.

        """
        data = load_model(fname=fname)
        if id:
            self.data = data.loc[data.loc[:, "GAF"] == id]
        else:
            self.data = data
        self._create_model_structure()

    def _create_model_structure(self):
        """Method to import model structure description from a file or
        database.

        """
        for id in self.data.loc[:, "EAG"].unique():
            df = self.data.loc[self.data.loc[:, "EAG"] == id]
            subpolder = SubPolder(id=id, polder=self, data=df)
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
