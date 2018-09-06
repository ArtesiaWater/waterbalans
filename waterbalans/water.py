from abc import ABC

import numpy as np
import pandas as pd

from .utils import makkink_to_penman


class WaterBase(ABC):
    __doc__ = """Base class from which all bucket classes inherit.

    """

    def __init__(self, id=None, eag=None, series=None, area=0.0):
        self.eag = eag  # Reference to mother object.

        self.series = pd.DataFrame()
        self.series = self.series.append(series)
        self.load_series_from_eag()

        self.parameters = pd.DataFrame(columns=["Waarde"])
        self.area = area  # area in square meters

    def initialize(self, tmin=None, tmax=None):
        """Method to initialize a Bucket with a clean DataFrame for the
        fluxes and storage time series. This method is called by the init
        and simulate methods.

        """
        if tmin is None:
            tmin = self.series.index.min()
        else:
            tmin = pd.Timestamp(tmin)

        if tmax is None:
            tmax = self.series.index.max()
        else:
            tmax = pd.Timestamp(tmax)

        index = self.series[tmin:tmax].index
        self.fluxes = pd.DataFrame(index=index, dtype=float)
        self.storage = pd.DataFrame(index=index, dtype=float)

    def load_series_from_eag(self):
        if self.eag is None:
            return

        if "Neerslag" in self.eag.series.columns:
            self.series["Neerslag"] = self.eag.series["Neerslag"]
        if "Verdamping" in self.eag.series.columns:
            self.series["Verdamping"] = self.eag.series["Verdamping"]

    def simulate(self, parameters, tmin=None, tmax=None, dt=1.0):
        pass

    def validate(self):
        """Method to validate the water balance based on the total input,
        output and the change in storage of the model for each time step.

        Returns
        -------

        """
        pass


class Water(WaterBase):
    """Water bucket used in the Eag class.

    The Water bucket is where all fluxes of the other buckets come together.

    Parameters
    ----------
    id: int
        id of the waterbucket.
    eag: waterbalans.Eag
        The eag the water bucket belongs to.
    series: list of pandas.Series or pandas.DataFrame
        ??? Not yet sure how this is gonna work.

    """

    def __init__(self, id, eag, series, area=0.0):
        WaterBase.__init__(self, id, eag, series, area)
        self.id = id
        self.eag = eag
        self.name = "Water"

        self.parameters = pd.DataFrame(
            data=[0, 0, 0, 0, 0, 0],
            index=['hTarget_1', 'hMin_1', 'hMax_1',
                   'hBottom_1', 'QInMax_1', 'QOutMax_1'],
            columns=["Waarde"])

        self.eag.add_water(self)

    def simulate(self, params=None, tmin=None, tmax=None, dt=1.0):
        self.initialize(tmin=tmin, tmax=tmax)

        # Get parameters
        self.parameters.update(params)
        hTarget_1, hMin_1, hMax_1, hBottom_1, QInMax_1, QOutMax_1 = \
            self.parameters.loc[:, "Waarde"]

        # 1. Add incoming fluxes from other buckets
        for bucket in self.eag.buckets.values():
            names = ["q_ui", "q_oa", "q_dr"]
            names = [name for name in names if name in bucket.fluxes.columns]
            fluxes = bucket.fluxes.loc[:, names] * -bucket.area
            fluxes.columns = [name + "_" + str(bucket.id) for name in names]
            self.fluxes = self.fluxes.join(fluxes, how="outer")

        # 2. calculate water bucket specific fluxes
        series = self.series.multiply(self.area)
        # TODO add series to fluxes without knowing the amount of series up front
        series.loc[:, "e"] = -makkink_to_penman(series.loc[:, "Verdamping"])
        self.fluxes = self.fluxes.join(series, how="outer")

        hMin_1 = hMin_1 * self.area
        hMax_1 = hMax_1 * self.area

        h = [hTarget_1 * self.area]
        q_in = []
        q_out = []

        q_totals = self.fluxes.sum(axis=1)

        # 3. Calculate the fluxes coming in and going out.
        for q_tot in q_totals.values:
            # Calculate the outgoing flux
            if h[-1] + q_tot > hMax_1:
                q_out.append(min(QOutMax_1, hMax_1 - h[-1] - q_tot))
                q_in.append(0.0)
            elif h[-1] + q_tot < hMin_1:
                q_in.append(hMin_1 - h[-1] - q_tot)
                q_out.append(0.0)
            else:
                q_out.append(0.0)
                q_in.append(0.0)

            h.append(h[-1] + q_in[-1] + q_out[-1] + q_tot)

        self.storage = pd.Series(data=np.array(h[1:]) - hBottom_1 * self.area,
                                 index=self.fluxes.index)
        self.level = pd.Series(data=np.array(h[1:]) / self.area,
                               index=self.fluxes.index)
        self.fluxes = self.fluxes.assign(q_in=q_in, q_out=q_out)
