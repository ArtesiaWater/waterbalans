"""Dit bestand bevat de EAG model klasse.

Raoul Collenteur, Artesia Water, September 2018
David Brakenhoff, Artesia Water, September 2018

"""

from collections import OrderedDict

import numpy as np
import pandas as pd
from pandas import Timestamp, date_range

from .buckets import Drain, MengRiool, Onverhard, Verhard
from .plots import Eag_Plots
from .timeseries import get_series


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

        # Container for all the buckets
        self.buckets = OrderedDict()
        # Eag attribute containing the water bucket
        self.water = None

        # This will be for future use when series are provided.
        if series is None:
            self.series = pd.DataFrame()
        else:
            self.series = series

        self.parameters = pd.DataFrame(columns=["Waarde"])

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
        """Adds a water bucket to the model. This is the bucket where all
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

    def add_series(self, series, tmin="2000", tmax="2015", freq="D", fillna=False):
        """Method to add timeseries based on a pandas DataFrame.

        Parameters
        ----------
        series: pandas.DataFrame
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        freq: str

        Notes
        -----

        Examples
        --------
        series = pd.read_csv("data\\reeks_16088_2501-EAG-1.csv", delimiter=";",
                      decimal=",")
        eag.add_series(series)

        """
        for id, df in series.groupby(["BakjeID", "ClusterType", "ParamType"]):
            BakjeID, ClusterType, ParamType = id
            # check if ValueSeries actually contains information
            if ParamType == "ValueSeries":
                if df.loc[:, "Waarde"].sum() == 0.0:
                    continue
            series = get_series(ClusterType, ParamType, df, tmin, tmax, freq)
            if fillna:
                if (series.isna().sum() > 0).all():
                    print("Filled {} NaN-values with 0.0 in series {}.".format(
                        np.int(series.isna().sum()), ClusterType))
                    series = series.fillna(0.0)

            if BakjeID in self.buckets.keys():
                self.buckets[BakjeID].series[ClusterType] = series
            elif BakjeID == self.water.id:
                if ClusterType.startswith("Cl"):
                    self.water.chloride[ClusterType] = series
                elif ClusterType.startswith("hTarget"):
                    self.water.hTargetSeries[ClusterType] = series
                else:
                    self.water.series[ClusterType] = series
            elif BakjeID == -9999:
                self.series[ClusterType] = series

        # Create index based on tmin/tmax if no series are added to EAG!
        if self.series.empty:
            self.series = pd.DataFrame(index=date_range(Timestamp(tmin),
                                                        Timestamp(tmax), freq="D"))

    def add_timeseries(self, series, name=None, tmin="2000", tmax="2015", freq="D",
                       fillna=False, method=None):
        """Method to add series directly to EAG. Series must contain volumes (so 
        not divided by area). Series must be negative for water taken out of the 
        EAG and positive for water coming into the EAG.

        Parameters
        ----------
        series: pandas.DataFrame or pandas.Series
        name: str, default None
            name of series to add, if not provided uses 
            first column name in DataFrame or Series name
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        freq: str

        """
        if self.series.index.shape[0] == 0:
            self.series = pd.DataFrame(index=date_range(Timestamp(tmin),
                                                        Timestamp(tmax), freq="D"))

        if name is None:
            if isinstance(series, pd.DataFrame):
                name = series.columns[0]
            elif isinstance(series, pd.Series):
                name = series.name

        if name in self.series.columns:
            print(
                "Warning! Series {} already present in EAG, overwriting data where not NaN!".format(name))
            first_valid_index = series.first_valid_index()
            last_valid_index = series.last_valid_index()
            series = series.loc[first_valid_index:last_valid_index].dropna()
            fillna = False

        if fillna:
            if (series.isna().sum() > 0).all():
                print("Filled {0} NaN-values with '{1}' in series {2}.".format(
                    np.int(series.isna().sum()), method, name))
                if isinstance(method, str):
                    series = series.fillna(method=method)
                elif isinstance(method, float) or isinstance(method, int):
                    series = series.fillna(method)

        shared_index = series.index.intersection(self.series.index)
        self.series.loc[shared_index, name] = series.loc[shared_index].values.squeeze()

    def get_series_from_gaf(self):
        """Load series from the Gaf instance if present and no series are
        provided.

        """
        # create index if empty
        if self.series.index.shape[0] == 0:
            tmin = self.gaf.series.index[0]
            tmax = self.gaf.series.index[-1]
            self.series = pd.DataFrame(index=date_range(Timestamp(tmin),
                                                        Timestamp(tmax), freq="D"))

        if self.gaf is not None and self.series.empty:
            self.series = self.series.join(self.gaf.series, how="left")

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
        print("Simulating: {}...".format(self.name))
        self.parameters = params
        self.parameters.set_index(self.parameters.loc[:, "Code"] + "_" +
                                  self.parameters.loc[:,
                                                      "Laagvolgorde"].astype(str), inplace=True)

        for id, bucket in self.buckets.items():
            p = params.loc[params.loc[:, "BakjeID"] == id]

            print("Simulating the waterbalance for bucket: %s %s" %
                  (bucket.name, id))
            bucket.simulate(params=p.loc[:, "Waarde"], tmin=tmin, tmax=tmax)

        p = params.loc[params.loc[:, "BakjeID"] == self.water.id]
        print("Simulating the waterbalance for bucket: %s %s" %
              (self.water.name, self.water.id))
        self.water.simulate(params=p.loc[:, "Waarde"], tmin=tmin, tmax=tmax)
        print("Simulation succesfully completed.")

    def aggregate_fluxes(self):
        """Method to aggregate fluxes to those used for visualisation in the
        Excel Waterbalance.

        Returns
        -------
        fluxes: pandas.DataFrame
            Pandas DataFrame with the fluxes. The column names denote the
            fluxes.

        """
        d = {
            "Neerslag": "neerslag",
            "Verdamping": "verdamping",
            "Qkwel": "kwel",
            "Qwegz": "wegzijging",
            "q_oa": "verhard",  # Verhard: q_oa van Verhard bakje
            "q_in": "berekende inlaat",
            "q_out": "berekende uitlaat",
            "q_dr": "drain",
            "Uitlaat": "uitlaat",
            "Inlaat": "inlaat"
        }

        fluxes = self.water.fluxes.reindex(columns=d.keys())
        parsed_cols = fluxes.dropna(how="all", axis=1).columns.tolist()
        fluxes = fluxes.rename(columns=d)

        # Verhard: q_oa van alle Verhard bakjes
        names = ["q_oa_" + str(id) for id in self.buckets.keys() if
                 self.buckets[id].name == "Verhard"]
        q_verhard = self.water.fluxes.loc[:, names]
        fluxes["verhard"] = q_verhard.sum(axis=1)

        # Uitspoeling: alle positieve q_ui fluxes uit alle verhard en onverhard en drain
        names = ["q_ui_" + str(id) for id in self.buckets.keys() if
                 self.buckets[id].name in ["Verhard", "Onverhard"]]
        q_uitspoel = self.water.fluxes.loc[:, names]
        q_uitspoel[q_uitspoel < 0] = 0
        fluxes["uitspoeling"] = q_uitspoel.sum(axis=1)

        # Intrek: alle negatieve q_ui fluxes uit alle bakjes behalve MengRiool
        names = ["q_ui_" + str(id) for id in self.buckets.keys() if
                 self.buckets[id].name != "MengRiool"]
        q_intrek = self.water.fluxes.loc[:, names]
        q_intrek[q_intrek > 0] = 0
        fluxes["intrek"] = q_intrek.sum(axis=1)

        # Oppervlakkige afstroming: q_oa van Onverharde en Drain bakjes
        names = ["q_oa_" + str(id) for id in self.buckets.keys() if
                 self.buckets[id].name in ["Onverhard", "Drain"]]
        q_afstroom = self.water.fluxes.loc[:, names]
        fluxes["afstroming"] = q_afstroom.sum(axis=1)

        # Combined Sewer Overflow: q_cso van MengRiool bakjes
        names = ["q_cso_" + str(id) for id in self.buckets.keys() if
                 self.buckets[id].name == "MengRiool"]
        q_cso = self.water.fluxes.loc[:, names]
        fluxes["q_cso"] = q_cso.sum(axis=1)

        # Gedraineerd: q_oa - positieve q_ui van Drain
        names = ["q_dr_" + str(id) for id in self.buckets.keys() if
                 self.buckets[id].name == "Drain"]
        names2 = ["q_ui_" + str(id) for id in self.buckets.keys() if
                  self.buckets[id].name == "Drain"]
        q_drain = self.water.fluxes.loc[:, names]
        q_uitspoeling = self.water.fluxes.loc[:, names2]
        q_uitspoeling[q_uitspoeling < 0.] = 0.
        fluxes["drain"] = q_drain.sum(axis=1) + q_uitspoeling.sum(axis=1)

        # Berekende in en uitlaat
        fluxes["berekende inlaat"] = self.water.fluxes["q_in"]
        fluxes["berekende uitlaat"] = self.water.fluxes["q_out"]

        # overige fluxes
        parsed_cols += ["q_in", "q_out"]
        for icol in self.water.fluxes.columns:
            if icol.startswith("q_"):
                parsed_cols.append(icol)
        missed_cols = self.water.fluxes.columns.difference(parsed_cols)
        for icol in missed_cols:
            fluxes[icol] = self.water.fluxes[icol]

        return fluxes

    def aggregate_with_pumpstation(self):
        fluxes = self.aggregate_fluxes()
        if "Gemaal" not in self.series.columns:
            print("Warning! No timeseries for pumping station. Cannot aggregate.")
            return fluxes
        fluxes.rename(columns={"berekende uitlaat": "sluitfout"}, inplace=True)
        # Add pumping station timeseries to fluxes
        fluxes["maalstaat"] = -1*self.series["Gemaal"]
        # Calculate difference between calculated and measured pumped volume
        fluxes["sluitfout"] = fluxes["sluitfout"].subtract(fluxes["maalstaat"])
        # Correct inlet volume with difference between calculated and measured
        # fluxes["berekende inlaat"] = fluxes["berekende inlaat"] - fluxes.loc[fluxes.sluitfout<0, "sluitfout"]

        return fluxes

    def calculate_chloride_concentration(self, params=None):
        """Calculate the chloride concentratation in the water bucket.

        Parameters
        ----------
        params: pandas.DataFrame
            Pandas DataFrame containing the parameters, similar to the
            simulate method.

        Returns
        -------
        C: pandas.Series
            Pandas Series of the chloride concentration in the Water bucket.

        """
        fluxes = self.aggregate_fluxes()
        self.mass_cl_tot = pd.Series(index=fluxes.index)

        # TODO: maybe these should be in params file?
        C_params = {
            "neerslag": 6,
            "kwel": 400,
            "verhard": 10,
            "drain": 70,
            "uitspoeling": 70,
            "afstroming": 35,
            "berekende inlaat": 100}

        C_params = pd.Series(C_params)

        # pick up ClInit from params
        if params is not None:
            Cl_init = params.loc[:, "Waarde"].iloc[0]
        else:
            Cl_init = 90.
            print(
                "Warning! Cl_init not in parameters, Setting default concentration to {0:.1f} mg/L".format(Cl_init))

        # pick up other Cl concentrations from water series
        rename_keys = {"ClVerhard": "verhard",
                       "ClRiolering": "q_cso",
                       "ClDrain": "drain",
                       "ClUitspoeling": "uitspoeling",
                       "ClAfstroming": "afstroming"}

        if not self.water.chloride.empty:
            cl_series = self.water.chloride.loc[:, rename_keys.keys()]
            cl_series = cl_series.rename(columns=rename_keys)
        else:
            cl_series = pd.DataFrame(
                index=self.series.index, columns=C_params.keys(), data=C_params.to_dict())

        for icol in set(C_params.index) - set(cl_series.columns):
            print("Warning! Setting default concentration of {0:.1f} mg/L for '{1}'!".format(
                C_params.loc[icol], icol))
            cl_series[icol] = C_params.loc[icol]

        # Bereken de initiele chloride massa
        hTarget = self.parameters.loc[self.parameters.loc[:, "Code"] ==
                                      "hTarget", "Waarde"].values[0]
        hBottom = self.parameters.loc[self.parameters.loc[:, "Code"] ==
                                      "hBottom", "Waarde"].values[0]

        V_init = (hTarget - hBottom) * self.water.area
        M = Cl_init * V_init
        C_out = Cl_init

        # Som van de uitgaande fluxen: wegzijging, intrek, berekende uitlaat
        V_out = fluxes.loc[:, [
            "intrek", "berekende uitlaat", "wegzijging"]].sum(axis=1)

        self.mass_cl_in = fluxes.loc[:, cl_series.columns].multiply(
            cl_series.loc[:, cl_series.columns])
        self.mass_cl_out = pd.Series(index=fluxes.index)
        for t in fluxes.index:
            M_in = self.mass_cl_in.loc[t].sum()

            M_out = V_out.loc[t] * C_out
            self.mass_cl_out.loc[t] = M_out

            M = M + M_in + M_out

            self.mass_cl_tot.loc[t] = M
            C_out = M / self.water.storage.loc[t, "storage"]

        C = self.mass_cl_tot / self.water.storage.storage

        return C

    def calculate_fractions(self):
        """Method to calculate the fractions.

        Returns
        -------
        frac: pandas.DataFrame
            pandas DataFrame with the fractions.

        """

        # TODO: Figure out how to include series with non-default names?
        fluxes = self.aggregate_fluxes()

        # TODO: check robustness of this solution, hard-coded was: ("verdamping", "wegzijging", "intrek", "berekende uitlaat")
        out_columns = fluxes.loc[:, fluxes.mean() < 0].columns
        outflux = fluxes.loc[:, out_columns].sum(axis=1)

        # TODO: check robustness of following solution, then remove hardcoded stuff:
        # fraction_columns = ["neerslag", "kwel", "verhard", "q_cso",
        #                     "drain", "uitspoeling", "afstroming", "berekende inlaat"]
        fraction_columns = fluxes.loc[:, fluxes.mean() > 0].columns

        fractions = pd.DataFrame(index=fluxes.index, columns=fraction_columns)
        # add starting day
        fractions.loc[fluxes.index[0] -
                      pd.Timedelta(days=1), fraction_columns] = 0.0
        # add starting day
        fractions.loc[fluxes.index[0]-pd.Timedelta(days=1), "initial"] = 1.0
        fractions.sort_index(inplace=True)

        for t in fluxes.index:
            fractions.loc[t, "initial"] = (fractions.loc[t - pd.Timedelta(days=1), "initial"] *
                                           self.water.storage.loc[t - pd.Timedelta(days=1), "storage"] +
                                           fractions.loc[t - pd.Timedelta(days=1), "initial"] *
                                           outflux.loc[t]) / self.water.storage.loc[t, "storage"]
            for icol in fraction_columns:
                fractions.loc[t, icol] = (fractions.loc[t - pd.Timedelta(days=1), icol] *
                                          self.water.storage.loc[t - pd.Timedelta(days=1), "storage"] +
                                          fluxes.loc[t, icol] -
                                          fractions.loc[t - pd.Timedelta(days=1), icol] *
                                          -1*outflux.loc[t]) / self.water.storage.loc[t, "storage"]

        return fractions

    def get_bucket_params(self):
        bucketnames = [b.name for b in self.buckets.values()]
        bucketparams = [b.parameters for b in self.buckets.values()]
        return pd.concat(bucketparams, axis=1, sort=False, keys=bucketnames)

    def get_buckets(self, buckettype="all"):
        if buckettype == "all":
            return list(self.buckets.values())
        else:
            bucketlist = []
            for _, v in self.buckets.items():
                if v.name == buckettype:
                    bucketlist.append(v)
            return bucketlist

    def modelstructure(self):
        df = pd.DataFrame(index=[i.id for i in self.buckets.values()])
        df.index.name = "ID"
        df["Name"] = [i.name for i in self.buckets.values()]
        df["Area"] = [i.area for i in self.buckets.values()]
        df["BucketObj"] = self.buckets.values()
        df.loc[self.water.id, :] = [self.water.name, self.water.area, self.water]
        return df
