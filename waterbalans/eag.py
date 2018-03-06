"""Dit bestand bevat de EAG model klasse.

Raoul Collenteur, Artesia Water, September 2017

"""

from collections import OrderedDict

from .buckets import *
from .plots import Eag_Plots


class Eag:
    __doc__ = """This class represents a EAG.
    """

    def __init__(self, gaf, name, series=None):
        # Basic information
        self.gaf = gaf
        self.name = name

        # Add some data
        self.buckets = OrderedDict()

        if series is None:
            self.series = pd.DataFrame()
        else:
            self.series = series

        self.parameters = pd.DataFrame(columns=["bucket", "pname", "pinit",
                                                "popt", "pmin", "pmax",
                                                "pvary"])

        # Add functionality from other modules
        self.plot = Eag_Plots(self)

    def add_bucket(self, bucket, replace=False):
        """Add a single bucket to the Eag.

        Parameters
        ----------
        bucket: waterbalans.BucketBase instance
            Bucket instance added to the model
        replace: bool
            Replace a bucket if a bucket with this name already exists

        """
        self.buckets[bucket.name] = bucket

    def add_water(self, water, replace=False):
        """

        Parameters
        ----------
        water: waterbalans WaterBase instance

        replace: bool
            force replace of the water object.

        """
        self.water = water

    def load_series(self):
        self.series["prec"] = self.gaf.series["prec"]
        self.series["evap"] = self.gaf.series["prec"]

    def get_init_parameters(self):
        """Method to obtain the parameters from the Buckets

        Returns
        -------

        """
        parameters = self.parameters
        for name, bucket in self.buckets.items():
            p = bucket.parameters
            p.loc[:, "bucket"] = name
            parameters = parameters.append(p, ignore_index=True)

        return parameters

    def get_parameters(self):
        pass

    def simulate(self, parameters=None, tmin=None, tmax=None):
        """Method to validate the water balance based on the total input,
        output and the change in storage of the model for each time step.

        Returns
        -------

        """
        if parameters is None:
            parameters = self.get_init_parameters()

        for name, bucket in self.buckets.items():
            p = parameters.loc[parameters.bucket == name, "popt"]

            print("Simulating the waterbalance for bucket: %s" % name)
            bucket.simulate(parameters=p.values, tmin=tmin, tmax=tmax)
