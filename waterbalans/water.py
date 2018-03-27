from abc import ABC

import pandas as pd

from .utils import makkink_to_penman


class WaterBase(ABC):
    __doc__ = """Base class from which all bucket classes inherit.

    """

    def __init__(self, id=None, eag=None, series=None, area=0.0):
        self.eag = eag  # Reference to mother object.
        self.series = pd.DataFrame()

        if series is None:
            self.load_series_from_eag()
        else:
            self.series = series

        self.parameters = pd.DataFrame(columns=["bucket", "pname", "pinit",
                                                "popt", "pmin", "pmax",
                                                "pvary"])
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
        series = dict()
        series["prec"] = self.eag.series["prec"]
        series["evap"] = self.eag.series["evap"]

        for name in ["seepage"]:
            pass
            # series[name] = load_series(name)

        return series

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
    def __init__(self, id, eag, series, area=0.0):
        WaterBase.__init__(self, id, eag, series, area)
        self.id = id
        self.eag = eag

        self.name = "Water"

        self.parameters = pd.DataFrame(index=["h_eq", "h_min", "h_max",
                                              "q_max"],
                                       columns=["pname", "pinit", "popt",
                                                "pmin", "pmax", "pvary"])
        self.parameters.loc[:, "pname"] = self.parameters.index

    def simulate(self, parameters=None, tmin=None, tmax=None, dt=1.0):
        self.initialize(tmin=tmin, tmax=tmax)

        if parameters is None:
            parameters = self.parameters.loc[:, "popt"]

        hTarget, h_min, hMax, h_bottom, QInMax, QOutMax = \
            parameters.loc[['hTarget', 'h_min', 'hMax', 'h_bottom', 'QInMax',
                            'QOutMax']]

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
        series.loc[:, "e"] = -makkink_to_penman(series.loc[:, "e"])
        self.fluxes = self.fluxes.join(series, how="outer")

        h_min = h_min * self.area
        hMax = hMax * self.area

        h = [hTarget * self.area]
        q_in = []
        q_out = []

        q_totals = self.fluxes.sum(axis=1)

        # 3. Calculate the fluxes coming in and going out.
        for q_tot in q_totals.values:
            # Calculate the outgoing flux
            if h[-1] + q_tot > hMax:
                q_out.append(min(QOutMax, hMax - h[-1] - q_tot))
                q_in.append(0.0)
            elif h[-1] + q_tot < h_min:
                q_in.append(h_min - h[-1] - q_tot)
                q_out.append(0.0)
            else:
                q_out.append(0.0)
                q_in.append(0.0)

            h.append(h[-1] + q_in[-1] + q_out[-1] + q_tot)

        self.storage = pd.Series(data=h[1:], index=self.fluxes.index)
        self.fluxes = self.fluxes.assign(q_in=q_in, q_out=q_out)
