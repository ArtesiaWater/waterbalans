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

        self.chloride = pd.DataFrame()

    def initialize(self, tmin=None, tmax=None):
        """Method to initialize a Bucket with a clean DataFrame for the
        fluxes and storage time series. This method is called by the init
        and simulate methods.

        """
        self.load_series_from_eag()

        if tmin is None:
            tmin = self.series.index.min()
        else:
            tmin = pd.Timestamp(tmin)

        if tmax is None:
            tmax = self.series.index.max()
        else:
            tmax = pd.Timestamp(tmax)

        self.series = self.series.loc[tmin:tmax]

        index = self.series.loc[tmin:tmax].index
        index_w_day_before = pd.DatetimeIndex([index[0] - pd.Timedelta(days=1)]).union(index)
        
        self.fluxes = pd.DataFrame(index=index, dtype=float)
        self.storage = pd.DataFrame(index=index_w_day_before, dtype=float)


    def load_series_from_eag(self):
        if self.eag is None:
            return

        if "Neerslag" in self.eag.series.columns:
            self.series["Neerslag"] = self.eag.series["Neerslag"]
        if "Verdamping" in self.eag.series.columns:
            self.series["Verdamping"] = self.eag.series["Verdamping"]
        if "Inlaat" in self.eag.series.columns:
            self.series["Inlaat"] = self.eag.series["Inlaat"] / self.area
        if "Uitlaat" in self.eag.series.columns:
            self.series["Uitlaat"] = -self.eag.series["Uitlaat"] / self.area

        # add remaining series to water bucket.
        otherseries = set(self.eag.series.columns) - {"Neerslag", "Verdamping", "Inlaat", "Uitlaat", "Peil", "Gemaal"}
        for name in otherseries:
            self.series[name] = self.eag.series[name] / self.area

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
            index=['hTarget_1', 'hTargetMin_1', 'hTargetMax_1',
                   'hBottom_1', 'QInMax_1', 'QOutMax_1'],
            columns=["Waarde"])

        self.eag.add_water(self)

    def simulate(self, params=None, tmin=None, tmax=None, dt=1.0):
        self.initialize(tmin=tmin, tmax=tmax)

        # Get parameters
        self.parameters.update(params)
        hTarget_1, hTargetMin_1, hTargetMax_1, hBottom_1, QInMax_1, QOutMax_1 = \
            self.parameters.loc[:, "Waarde"]

        # 1. Add incoming fluxes from other buckets
        for bucket in self.eag.buckets.values():
            names = ["q_ui", "q_oa", "q_dr", "q_cso"]
            names = [name for name in names if name in bucket.fluxes.columns]
            fluxes = bucket.fluxes.loc[:, names] * -bucket.area
            fluxes.columns = [name + "_" + str(bucket.id) for name in names]
            self.fluxes = self.fluxes.join(fluxes, how="outer")

        # 2. calculate water bucket specific fluxes
        series = self.series.multiply(self.area)

        # TODO add series to fluxes without knowing the amount of series up front
        series.loc[:, "Verdamping"] = -makkink_to_penman(series.loc[:,"Verdamping"])
        series.loc[:, "Qwegz"] = -series.loc[:, "Qwegz"]
        self.fluxes = self.fluxes.join(series, how="outer")

        hTargetMin_1 = (hTarget_1 - hTargetMin_1 - hBottom_1) * self.area
        hTargetMax_1 = (hTargetMax_1 + hTarget_1 - hBottom_1) * self.area

        if "Peil" in self.eag.series.loc[tmin:tmax].dropna(axis=1, how="all").columns:
            h = (self.eag.series.loc[tmin:tmax, "Peil"] - hBottom_1) * self.area
            if np.isnan(h.iloc[0]):
                # if no first value, calculate one based on hTarget:
                h.loc[h.index[0]-pd.Timedelta(days=1)] = (hTarget_1 - hBottom_1) * self.area
                h.sort_index(inplace=True)
        else:
            h = pd.Series(index=self.eag.series.loc[pd.Timestamp(tmin) - 
                          pd.Timedelta(days=1):pd.Timestamp(tmax)].index)
            h.iloc[0] = (hTarget_1 - hBottom_1) * self.area
        q_in = pd.Series(index=self.eag.series.loc[tmin:tmax].index, 
                         data=np.zeros(self.eag.series.loc[tmin:tmax].shape[0]))
        q_out = pd.Series(index=self.eag.series.loc[tmin:tmax].index, 
                          data=np.zeros(self.eag.series.loc[tmin:tmax].shape[0]))
        
        q_totals = self.fluxes.sum(axis=1)
        
        for t in h.index[1:]:
            if ~np.isnan(h.loc[t]):  # there is a water level measurement
                dh_minus_dq = h.loc[t] - h.loc[t-pd.Timedelta(days=1)] - q_totals.loc[t]
                if dh_minus_dq > 0.0:
                    q_in.loc[t] = dh_minus_dq
                elif dh_minus_dq < 0.0:
                    q_out.loc[t] = dh_minus_dq

            else:  # no water level measurement
                if h.loc[t - pd.Timedelta(days=1)] + q_totals.loc[t] > hTargetMax_1:
                    q_out.loc[t] = min(QOutMax_1, hTargetMax_1 - h[t - pd.Timedelta(days=1)] - q_totals.loc[t])
                elif h[t - pd.Timedelta(days=1)] + q_totals.loc[t] < hTargetMin_1:
                    q_in.loc[t] = hTargetMin_1 - h[t - pd.Timedelta(days=1)] - q_totals.loc[t]
                
                h.loc[t] = h.loc[t - pd.Timedelta(days=1)] + q_in.loc[t] + q_out.loc[t] + q_totals.loc[t]

        self.storage = self.storage.assign(storage=h)
        self.level = pd.Series(data=h.values / self.area + hBottom_1,
                               index=h.index)
        self.fluxes = self.fluxes.assign(q_in=q_in, q_out=q_out)

    def validate(self, return_wb_series=False):
        """Method to validate the water balance based on the total input,
        output and the change in storage of the model for each time step.

        Returns
        -------

        """
        if not hasattr(self, "fluxes"):
            raise AttributeError("No attribute 'fluxes'. Run simulate first")
        
        wb = pd.DataFrame(index=self.fluxes.index, columns=["DeltaS", "DeltaQ"])

        wb.loc[:, "DeltaS"] = self.storage.diff(periods=1)
        wb.loc[:, "DeltaQ"] = self.fluxes.sum(axis=1)

        wb["Water balance"] = wb.loc[:, "DeltaS"] - wb.loc[:, "DeltaQ"]

        if return_wb_series:
            return wb
        else:
            return np.allclose(wb["Water balance"].dropna(), 0.0)
