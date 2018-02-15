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

        self.parameters = pd.DataFrame(columns=["pinit", "popt", "pmin",
                                                "pmax", "pvary"])

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

    def _load_buckets(self, data):
        """Method to load the buckets for the subpolder.

        """
        for kind in data.loc[:, "TYPE_WBAL"].values:
            df = self.data.loc[self.data.loc[:, "TYPE_WBAL"] == kind]
            bucket = Bucket(kind=kind, polder=self, data=df)
            self.buckets[kind] = bucket

    def load_series(self):
        self.series["prec"] = self.gaf.series["prec"]
        self.series["evap"] = self.gaf.series["prec"]

    def get_init_parameters(self):
        pass

    def get_parameters(self):
        pass

    def simulate(self, tmin=None, tmax=None):
        """Method to validate the water balance based on the total input,
        output and the change in storage of the model for each time step.

        Returns
        -------

        """
        for bucket in self.buckets.values():
            print("Simulating the waterbalance for bucket: %s" % bucket.name)
            bucket.simulate(tmin=tmin, tmax=tmax)
