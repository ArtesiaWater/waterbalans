"""Dit bestand bevat de EAG model klasse.

Raoul Collenteur, Artesia Water, September 2017

"""

from .buckets import *
from collections import OrderedDict


class SubPolder:
    def __init__(self, id, polder, data):
        self.data = data  # Pandas dataframe with the model table
        self.id = id
        self.polder = polder  # Reference to the mother object

        self.buckets = OrderedDict()
        self._load_buckets(data)

        self.series = OrderedDict()
        # self.load_series()

    def _load_buckets(self, data):
        """Method to load the buckets for the subpolder.

        """
        for kind in data.loc[:, "TYPE_WBAL"].values:
            df = self.data.loc[self.data.loc[:, "TYPE_WBAL"] == kind]
            bucket = Bucket(kind=kind, polder=self, data=df)
            self.buckets[kind] = bucket

    def load_series(self):
        self.series["prec"] = self.polder.series["prec"]
        self.series["evap"] = self.polder.series["prec"]

    def calculate_wb(self):
        pass

    def validate_wb(self):
        """Method to validate the water balance based on the total input,
        output and the change in storage of the model for each time step.

        Returns
        -------

        """
        pass
