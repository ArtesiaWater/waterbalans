"""Dit bestand bevat de EAG model klasse.

Raoul Collenteur, Artesia Water, September 2017

"""

from collections import OrderedDict

from .buckets import *
from .plots import Eag_Plots


class Eag:
    """This class represents an EAG.

    Parameters
    ----------
    id: int, optional
        integer with the id of the EAG.
    name: str
        String wiuth the name of the Eag.
    gaf: waterbalans.Gaf, optional
        Instance of a Gaf waterbalans

    Notes
    -----
    The Eag class can be used on its own, without the use of a Gaf instance.
    As such, the waterbalance for an Eag can be calculated stand alone.

    """

    def __init__(self, id=None, name=None, gaf=None, series=None):
        # Basic information
        self.gaf = gaf
        self.id = id
        self.name = name

        # Add some data
        self.buckets = OrderedDict()
        self.water = None

        # This will be for future use when series are provided.
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
        if bucket.id in self.buckets.keys() and replace is False:
            raise KeyError("bucket with id %s is already in buckets dict."
                           % bucket.id)
        else:
            self.buckets[bucket.id] = bucket

    def add_water(self, water, replace=False):
        """Adds a water bucket to the model. This is the "place" where all
        fluxes of an EAG come together.

        Parameters
        ----------
        water: waterbalans.WaterBase instance
            Instance of the WaterBase class.
        replace: bool
            force replace of the water object.

        """
        if self.water is not None and replace is False:
            raise KeyError("There is already a water bucket present in the "
                           "model.")
        else:
            self.water = water

    def load_series_from_gaf(self):
        """Load series from the Gaf instance if present and no series are
        provided.

        """
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
            p.loc[:, "BakjeID"] = name
            parameters = parameters.append(p, ignore_index=True)

        return parameters

    def simulate(self, params, tmin=None, tmax=None):
        """Method to validate the water balance based on the total input,
        output and the change in storage of the model for each time step.

        Parameters
        ----------
        params: pd.DataFrame
            Pandas DataFrame with the parameters.
        tmin: str or pandas.Timestamp
        tmax: str or pandas.Timestamp

        """

        for id, bucket in self.buckets.items():
            p = params.loc[params.loc[:, "Bakjes_ID"] == id]
            p.set_index(p.loc[:, "Code"] + "_" +
                        p.loc[:, "LaagVolgorde"].astype(str), inplace=True)

            print("Simulating the waterbalance for bucket: %s" % id)
            bucket.simulate(params=p.loc[:, "Waarde"], tmin=tmin, tmax=tmax)

        p = params.loc[params.loc[:, "Bakjes_ID"] == self.water.id]
        p.set_index(p.loc[:, "Code"] + "_" +
                    p.loc[:, "LaagVolgorde"].astype(str), inplace=True)
        self.water.simulate(params=p.loc[:, "Waarde"], tmin=tmin, tmax=tmax)

    def aggregate_fluxes(self):
        d = {
            "p": "neerslag",
            "e": "verdamping",
            "s": "kwel",
            "w": "wegzijging",
            "x": "inlaat",
            "q_out": "uitlaat",
            "q_oa_2": "verhard",  # Verhard: q_oa van Verhard bakje
        }

        fluxes = self.water.fluxes.loc[:, d.keys()]
        fluxes = fluxes.rename(columns=d)

        # Uitspoeling: alle positieve q_ui fluxes uit alle verhard en onverhard
        names = ["q_ui_" + str(id) for id in self.buckets.keys() if
                 self.buckets[id].name
                 in ["Verhard", "Onverhard"]]
        q_uitspoel = self.water.fluxes.loc[:, names]
        q_uitspoel[q_uitspoel < 0] = 0
        fluxes["uitspoeling"] = q_uitspoel.sum(axis=1)

        # Intrek: alle negatieve q_ui fluxes uit alle bakjes
        names = ["q_ui_" + str(id) for id in self.buckets.keys()]
        q_intrek = self.water.fluxes.loc[:, names]
        q_intrek[q_intrek > 0] = 0
        fluxes["intrek"] = q_intrek.sum(axis=1)

        # Oppervlakkige afstroming: q_oa van Onverharde bakjes
        names = ["q_oa_" + str(id) for id in self.buckets.keys() if
                 self.buckets[id].name
                 == "Onverhard"]
        q_afstroom = self.water.fluxes.loc[:, names]
        fluxes["afstroming"] = q_afstroom.sum(axis=1)

        # Gedraineerd: q_oa - positieve q_ui van Drain
        fluxes["drain"] = 0

        return fluxes
