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
        # TODO: needed here? Also called in initialize.
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
        index_w_day_before = pd.DatetimeIndex(
            [index[0] - pd.Timedelta(days=1)]).union(index)

        self.fluxes = pd.DataFrame(index=index, dtype=float)
        self.storage = pd.DataFrame(index=index_w_day_before, dtype=float)
        self.level = pd.DataFrame(index=index_w_day_before, dtype=float)

    def load_series_from_eag(self):
        if self.eag is None:
            return
        colset = []
        for icol in self.eag.series.columns:
            if icol.lower().startswith("neerslag"):
                colset.append(icol)
                self.series["Neerslag"] = self.eag.series[icol]
            if icol.lower().startswith("verdamping"):
                colset.append(icol)
                self.series["Verdamping"] = self.eag.series[icol]
            if icol.lower().startswith("inlaat"):
                colset.append(icol)
                self.series[icol] = self.eag.series[icol] / self.area
            if icol.lower().startswith("uitlaat"):
                colset.append(icol)
                self.series[icol] = -self.eag.series[icol] / self.area
            if icol.lower().startswith("gemaal"):
                colset.append(icol)

        # add remaining series to water bucket.
        colset += ["Neerslag", "Verdamping", "Peil", "Gemaal", "q_cso"]
        otherseries = set(self.eag.series.columns) - set(colset)

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

    def __init__(self, id, eag, series, area=0.0, use_waterlevel_series=False):
        WaterBase.__init__(self, id, eag, series, area)
        self.id = id
        self.eag = eag
        self.name = "Water"
        self.use_waterlevel_series = use_waterlevel_series

        self.parameters = pd.DataFrame(
            data=[0, 0, 0, 0, np.nan, np.nan],
            index=['hTarget_1', 'hTargetMin_1', 'hTargetMax_1',
                   'hBottom_1', 'QInMax_1', 'QOutMax_1'],
            columns=["Waarde"])

        self.hTargetSeries = pd.DataFrame()  # for setting waterlevel targets as series
        self.eag.add_water(self)
    
    def __repr__(self):
        return "<{0}: {1} bucket with area {2:.1f}>".format(self.id, "Water", self.area)

    def simulate(self, params=None, tmin=None, tmax=None, dt=1.0):
        self.initialize(tmin=tmin, tmax=tmax)

        # Get parameters
        self.parameters.update(params)
        hTarget_1, hTargetMin_1, hTargetMax_1, hBottom_1, QInMax_1, QOutMax_1 = \
            self.parameters.loc[:, "Waarde"]

        # Pick up hTargetSeries if they exist
        if not self.hTargetSeries.empty and not self.use_waterlevel_series:
            hTargetMin_1 = self.hTargetSeries["hTargetMin"]
            hTargetMax_1 = self.hTargetSeries["hTargetMax"]

        if QInMax_1 == 0.:
            print(
                "Warning! 'QInMax_1' is equal to 0. Assuming this means there is no limit to inflow.")
        if QOutMax_1 == 0.:
            print(
                "Warning! 'QOutMax_1' is equal to 0. Assuming this means there is no limit to outflow.")

        # 1. Add incoming fluxes from other buckets
        for bucket in self.eag.buckets.values():
            names = ["q_ui", "q_oa", "q_dr", "q_cso"]
            names = [name for name in names if name in bucket.fluxes.columns]
            fluxes = bucket.fluxes.loc[:, names] * -bucket.area
            fluxes.columns = [name + "_" + str(bucket.id) for name in names]
            self.fluxes = self.fluxes.join(fluxes, how="outer")

        # 2. calculate water bucket specific fluxes
        series = self.series.multiply(self.area)

        # Add series to fluxes without knowing the amount of series up front
        # TODO: change back to False!!
        series.loc[:, "Verdamping"] = - \
            makkink_to_penman(
                series.loc[:, "Verdamping"], use_excel_factors=True)
        if "Qwegz" in series.columns:
            series.loc[:, "Qwegz"] = -series.loc[:, "Qwegz"]
        self.fluxes = self.fluxes.join(series, how="outer")

        if self.use_waterlevel_series:
            h = (self.eag.series.loc[tmin:tmax,
                                     "Peil"] - hBottom_1) * self.area
            # TODO: currently set to start with target level regardless of observations. Check if correct for all EAGs.
            # starting level calculated based on hTarget
            h.loc[h.index[0]-pd.Timedelta(days=1)
                  ] = (hTarget_1 - hBottom_1) * self.area
            h.sort_index(inplace=True)
        else:
            h = pd.Series(index=self.storage.loc[pd.Timestamp(tmin) -
                                                 pd.Timedelta(days=1):pd.Timestamp(tmax)].index)
            h.iloc[0] = (hTarget_1 - hBottom_1) * self.area

        # pre-allocate empty series
        q_in = pd.Series(index=self.eag.series.loc[tmin:tmax].index,
                         data=np.zeros(self.eag.series.loc[tmin:tmax].shape[0]))
        q_out = pd.Series(index=self.eag.series.loc[tmin:tmax].index,
                          data=np.zeros(self.eag.series.loc[tmin:tmax].shape[0]))

        # net flux
        q_totals = self.fluxes.sum(axis=1)

        # Static/dynamic target levels:
        #   - Negative offsets will result in offset being set statically, 
        #     i.e. it will not vary over simulation
        #   - Positive offsets will result in offset being set dynamically 
        #     relative to observed level if available

        if not self.hTargetSeries.empty and not self.use_waterlevel_series:
            # Convert to volume:
            hTargetMin_1 = (hTargetMin_1 - hBottom_1) * self.area
            hTargetMax_1 = (hTargetMax_1 - hBottom_1) * self.area
        else:  # hTargets are floats (positive or negative) relative to Target level
            if hTargetMin_1 <= 0:  # static
                hTargetMin_static = (
                    hTarget_1 + hTargetMin_1 - hBottom_1) * self.area  # a volume
                hTargetMin_1 = pd.Series(index=h.index, data=hTargetMin_static)
            else:  # dynamic
                ht = hTargetMin_1
                hTargetMin_1 = (self.eag.series.loc[tmin:tmax, "Peil"] -
                                hTargetMin_1 - hBottom_1) * self.area
                # This is what Excel does (start with init level instead of first obs Peil)
                hTargetMin_1.iloc[0] = (hTarget_1 - ht - hBottom_1) * self.area

            if hTargetMax_1 <= 0:  # static
                hTargetMax_static = (
                    hTarget_1 - hTargetMax_1 - hBottom_1) * self.area  # a volume
                hTargetMax_1 = pd.Series(index=h.index, data=hTargetMax_static)
            else:  # dynamic
                ht = hTargetMax_1
                hTargetMax_1 = (self.eag.series.loc[tmin:tmax, "Peil"] +
                                hTargetMax_1 - hBottom_1) * self.area
                # This is what Excel does (start with init level instead of first obs Peil)
                hTargetMax_1.iloc[0] = (hTarget_1 + ht - hBottom_1) * self.area

        for t in h.index[1:]:
            if np.isnan(hTargetMax_1.loc[t]):
                hTargetMax_1.loc[t] = hTargetMax_1.loc[t - pd.Timedelta(days=1)]
            if np.isnan(hTargetMin_1.loc[t]):
                hTargetMin_1.loc[t] = hTargetMin_1.loc[t - pd.Timedelta(days=1)]
            
            hTargetMax_obs = hTargetMax_1.loc[t]
            hTargetMin_obs = hTargetMin_1.loc[t]

            if ~np.isnan(h.loc[t]):  # there is a water level measurement
                # volume[t] = volume[t-1] + q_net[t]
                h_plus_q = h.loc[t-pd.Timedelta(days=1)] + q_totals.loc[t]

                # test if new volume exceeds thresholds
                if h_plus_q > hTargetMax_obs:
                    if np.isnan(QOutMax_1) or (QOutMax_1 == 0):  # no limit on out flux
                        q_out.loc[t] = hTargetMax_obs - h_plus_q
                    else:  # limit on out flux
                        # No limit on outflux in first timestep in Excel
                        if t == tmin:
                            q_out.loc[t] = hTargetMax_obs - h_plus_q
                        else:
                            q_out.loc[t] = max(-1*QOutMax_1,
                                            hTargetMax_obs - h_plus_q)
                elif h_plus_q < hTargetMin_obs:
                    if np.isnan(QInMax_1) or (QInMax_1 == 0):  # no limit on in flux
                        q_in.loc[t] = hTargetMin_obs - h_plus_q
                    else:  # limit on in flux
                        q_in.loc[t] = min(QInMax_1, hTargetMin_obs - h_plus_q)

                # update h with new calculated volume
                h.loc[t] = h.loc[t - pd.Timedelta(days=1)] + \
                    q_in.loc[t] + q_out.loc[t] + q_totals.loc[t]

            else:  # no water level measurement
                # volume[t] = volume[t-1] + q_net[t]
                h_plus_q = h.loc[t-pd.Timedelta(days=1)] + q_totals.loc[t]

                if h_plus_q > hTargetMax_obs:
                    if np.isnan(QOutMax_1) or (QOutMax_1 == 0):
                        q_out.loc[t] = hTargetMax_obs - h_plus_q
                    else:
                        # No limit on outflux in first timestep in Excel
                        if t == tmin:
                            q_out.loc[t] = hTargetMax_obs - h_plus_q
                        else:
                            q_out.loc[t] = max(-1*QOutMax_1,
                                            hTargetMax_obs - h_plus_q)
                elif h_plus_q < hTargetMin_obs:
                    if np.isnan(QInMax_1) or (QInMax_1 == 0):
                        q_in.loc[t] = hTargetMin_obs - h_plus_q
                    else:
                        q_in.loc[t] = min(QInMax_1, hTargetMin_obs - h_plus_q)

                h.loc[t] = h.loc[t - pd.Timedelta(days=1)] + \
                    q_in.loc[t] + q_out.loc[t] + q_totals.loc[t]

        self.storage = self.storage.assign(storage=h)
        self.level = self.level.assign(level=h / self.area + hBottom_1)
        self.fluxes = self.fluxes.assign(q_in=q_in, q_out=q_out)

    def validate(self, return_wb_series=False):
        """Method to validate the water balance based on the total input,
        output and the change in storage of the model for each time step.

        Returns
        -------

        """
        if not hasattr(self, "fluxes"):
            raise AttributeError("No attribute 'fluxes'. Run simulate first")

        wb = pd.DataFrame(index=self.fluxes.index,
                          columns=["DeltaS", "DeltaQ"])

        wb.loc[:, "DeltaS"] = self.storage.diff(periods=1)
        wb.loc[:, "DeltaQ"] = self.fluxes.sum(axis=1)

        wb["Water balance"] = wb.loc[:, "DeltaS"] - wb.loc[:, "DeltaQ"]

        if return_wb_series:
            return wb
        else:
            return np.allclose(wb["Water balance"].dropna(), 0.0)
