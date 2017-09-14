"""Dit bestand bevat de EAG model klasse.

"""

from collections import OrderedDict

from .buckets import *


class SubPolder:
    def __init__(self, id, polder, data):
        self.id = id
        self.polder = polder  # Reference to the mother object

        self.buckets = OrderedDict()
        self.load_model(data)

        self.series = OrderedDict()
        self.load_series()

    def load_model(self, data):
        """Method to load the buckets for the subpolder.

        """

        for kind in data.keys():
            bucket = Bucket(self)
            self.buckets[kind] = bucket

        self.name = None

        self.x = 0.0
        self.y = 0.0
        self.area = 0.0

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
