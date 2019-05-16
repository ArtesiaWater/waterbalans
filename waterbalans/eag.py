"""Dit bestand bevat de EAG model klasse.

Raoul Collenteur, Artesia Water, September 2018
David Brakenhoff, Artesia Water, September 2018

"""

import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
from pandas import Timestamp, date_range
from pandas.tseries.offsets import MonthOffset

from .plots import Eag_Plots
from .timeseries import get_series, update_series


class Eag:
    """This class represents an EAG.

    Parameters
    ----------
    idn: int, optional
        integer with the ID of the EAG.
    name: str
        String wiuth the name of the Eag.
    gaf: waterbalans.Gaf, optional
        Instance of a Gaf waterbalans

    Notes
    -----
    The Eag class can be used on its own, without the use of a Gaf instance.
    As such, the waterbalance for an Eag can be calculated stand alone.

    """

    def __init__(self, idn=None, name=None, gaf=None, series=None):

        self.logger = self.get_logger()

        # Basic information
        self.gaf = gaf
        self.idn = idn
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

    def __repr__(self):
        return "<EAG object: {0}>".format(self.name)

    def get_logger(self, log_level=logging.INFO, filename=None):

        logging.basicConfig(format='%(asctime)s | %(funcName)s - %(levelname)s : %(message)s',
                            level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(log_level)

        if filename is not None:
            fhandler = logging.FileHandler(filename=filename, mode='w')
            logger.addHandler(fhandler)

        return logger

    def add_bucket(self, bucket, replace=False):
        """Add a single bucket to the Eag.

        Parameters
        ----------
        bucket: waterbalans.BucketBase instance
            Bucket instance added to the model
        replace: bool
            Replace a bucket if a bucket with this name already exists

        """
        if bucket.idn in self.buckets.keys() and replace is False:
            raise KeyError("bucket with ID %s is already in buckets dict."
                           % bucket.idn)
        else:
            self.buckets[bucket.idn] = bucket

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

    def add_series_from_database(self, series, tmin="2000", tmax="2015",
                                 freq="D", fillna=False, method="append"):
        """Method to add timeseries based on a DataFrame containing
        information about the series. Series are described by one or more
        rows in the DataFrame with at least the following columns:

         - Bucket ID: ID of the bucket the series should be added to
         - SeriesType (unfortunately called ParamType at the moment): Origin or Type
           of the Series: e.g. FEWS, KNMI, Local, ValueSeries, Constant
         - ClusterType: Name of the parameter

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
        # Sort series to parse in order: Valueseries -> Local -> FEWS -> Constant
        series = series.sort_values(by="ParamType", ascending=False)
        self.logger.info(
            "Parsing timeseries from database export and adding to EAG.")

        for idn, df in series.groupby(["ParamType", "BakjeID", "ClusterType"], sort=False):
            ParamType, BakjeID, ClusterType = idn
            series_list = []
            # check if ValueSeries actually contains information
            if ParamType == "ValueSeries" and df.loc[:, "Waarde"].sum() == 0.0:
                continue
            elif ParamType == "FEWS" and df.shape[0] > 1:
                # deal with multiple FEWS IDs for one ClusterType
                self.logger.warning(
                    "Multiple FEWS series found for {}.".format(ClusterType))
                for i in range(df.shape[0]):
                    series_list.append(get_series(
                        ClusterType, ParamType, df.iloc[i:i+1], tmin, tmax, freq))
            else:  # single series
                series_list.append(get_series(
                    ClusterType, ParamType, df, tmin, tmax, freq))

            for s in series_list:
                # Check if series contains data
                if s is None:
                    continue
                # Fill NaNs if specified
                if fillna:
                    if (s.isna().sum() > 0).all():
                        self.logger.info("Filled {} NaN-values with 0.0 in series {}.".format(
                            np.int(s.isna().sum()), ClusterType))
                        s = s.fillna(0.0)
                # Add series to appropriate object
                if BakjeID in self.buckets.keys():  # add to Land Bucket
                    # check if already exists
                    if ClusterType in self.buckets[BakjeID].series.columns:
                        orig_series = self.buckets[BakjeID].series[ClusterType]
                        new_series = update_series(
                            orig_series, s, method=method)
                        # add updated series
                        self.buckets[BakjeID].series[ClusterType] = new_series
                    else:  # add new series
                        self.buckets[BakjeID].series[ClusterType] = s
                elif BakjeID == self.water.idn:  # add to Water Bucket
                    if ClusterType.startswith("hTarget"):
                        self.water.hTargetSeries[ClusterType] = s
                    else:  # non-targetlevel series
                        if ClusterType in self.water.series.columns:  # check if already exists
                            orig_series = self.water.series[ClusterType]
                            new_series = update_series(
                                orig_series, s, method=method)
                            # add updated series
                            self.water.series[ClusterType] = new_series
                        else:  # add new series
                            self.water.series[ClusterType] = s
                elif BakjeID == -9999:  # add to EAG, no specific bucket defined
                    if ClusterType in self.series.columns:  # check if already exists
                        orig_series = self.series[ClusterType]
                        new_series = update_series(
                            orig_series, s, method=method)
                        if new_series.empty:
                            raise(ValueError("Empty series!"))
                        # add updated series
                        self.series[ClusterType] = new_series
                    else:  # add new series
                        self.series[ClusterType] = s
                else:
                    self.logger.warning(
                        "Series '{}' not added.".format(ClusterType))

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
            self.logger.info(
                "Adding timeseries '{0}' to EAG manually".format(name))
            self.logger.warning(
                "Series {} already present in EAG, overwriting data where not NaN!".format(name))
            first_valid_index = series.first_valid_index()
            last_valid_index = series.last_valid_index()
            series = series.loc[first_valid_index:last_valid_index].dropna()
            fillna = False

        if fillna:
            if (series.isna().sum() > 0).all():
                self.logger.info("Filled {0} NaN-values with '{1}' in series {2}.".format(
                    np.int(series.isna().sum()), method, name))
                if isinstance(method, str):
                    series = series.fillna(method=method)
                elif isinstance(method, float) or isinstance(method, int):
                    series = series.fillna(method)

        shared_index = series.index.intersection(self.series.index)
        self.series.loc[shared_index,
                        name] = series.loc[shared_index].values.squeeze()

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

    def get_modelstructure(self):
        df = pd.DataFrame(index=[i.idn for i in self.buckets.values()])
        df.index.name = "ID"
        df["Name"] = [i.name for i in self.buckets.values()]
        df["Area"] = [i.area for i in self.buckets.values()]
        df["BucketObj"] = self.buckets.values()
        df.loc[self.water.idn, :] = [
            self.water.name, self.water.area, self.water]
        return df

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
        self.logger.info("Simulating: {}...".format(self.name))
        self.parameters = params
        self.parameters.set_index(self.parameters.loc[:, "ParamCode"] + "_" +
                                  self.parameters.loc[:,
                                                      "Laagvolgorde"].astype(str), inplace=True)

        for idn, bucket in self.buckets.items():
            p = params.loc[params.loc[:, "BakjeID"] == idn]

            self.logger.info("Simulating the waterbalance for bucket: %s %s" %
                             (bucket.name, idn))
            bucket.simulate(params=p.loc[:, "Waarde"], tmin=tmin, tmax=tmax)

        p = params.loc[params.loc[:, "BakjeID"] == self.water.idn]
        self.logger.info("Simulating the waterbalance for bucket: %s %s" %
                         (self.water.name, self.water.idn))
        self.water.simulate(params=p.loc[:, "Waarde"], tmin=tmin, tmax=tmax)
        self.logger.info("Simulation succesfully completed.")

    def simulate_wq(self, wq_params, increment=False, tmin=None,
                    tmax=None, freq="D"):

        self.logger.info("Simulating water quality: {}...".format(self.name))

        if not hasattr(self.water, "fluxes"):
            raise AttributeError("No calculated fluxes in water bucket."
                                 "Please simulate water quantity first!")

        # Get tmin and tmax
        if tmin is None:
            tmin = self.series.index.min()
        else:
            tmin = pd.Timestamp(tmin)
        if tmax is None:
            tmax = self.series.index.max()
        else:
            tmax = pd.Timestamp(tmax)

        # Get fluxes
        fluxes = self.aggregate_fluxes()
        fluxes.columns = [icol.lower() for icol in fluxes.columns]

        # Parse wq_params table
        # Should result in C_series -> per flux a series in one DataFrame,
        # if FEWS or local data is used, data is ffilled
        incols = [icol.lower()
                  for icol in wq_params.Inlaattype if icol.lower() != "initieel"]
        incols = [i for i in incols if i in fluxes.columns]

        C_series = pd.DataFrame(
            index=self.water.fluxes.loc[tmin:tmax].index, columns=incols)

        C_init = 0.0  # if no initial value passed

        for ID, df in wq_params.groupby(["Inlaattype", "Reekstype"]):
            inlaat_type, reeks_type = ID
            inlaat_type = inlaat_type.lower()

            # Get initial concentration and continue
            if inlaat_type == "initieel":
                C_init = df["Waarde"].iloc[0]
                continue

            series = get_series(inlaat_type, reeks_type, df,
                                tmin=tmin, tmax=tmax, freq=freq)

            if series.sum() == 0.0:
                if inlaat_type in incols:
                    incols.remove(inlaat_type)
                continue

            # If increment is True, add increment to concentration
            if increment:
                series += df["Stofincrement"].iloc[0]

            # add series to C_series DataFrame
            if reeks_type == "Constant":
                C_series.loc[:, inlaat_type] = series
            else:
                # Fill in series on dates with measurement
                shared_index = series.index.intersection(C_series.index)
                C_series.loc[shared_index, inlaat_type] = series

            # Series is often not measured on each day -> ffill to fill gaps
            if series.isna().sum() > 0:
                C_series.loc[:, inlaat_type].fillna(method='ffill')

        # Calculate initial mass and concentration
        hTarget = self.parameters.loc[self.parameters.loc[:, "ParamCode"] ==
                                      "hTarget", "Waarde"].values[0]
        hBottom = self.parameters.loc[self.parameters.loc[:, "ParamCode"] ==
                                      "hBottom", "Waarde"].values[0]

        V_init = (hTarget - hBottom) * self.water.area
        M = C_init * V_init
        C_out = C_init

        # Sum of outgoing fluxes from water bucket
        outcols = ["intrek", "berekende uitlaat", "wegzijging"]
        outcols += [jcol.lower()
                    for jcol in self.water.fluxes if jcol.startswith("Uitlaat")]
        V_out = fluxes.loc[:, outcols]

        mass_tot = pd.Series(index=fluxes.index,
                             name="mass_tot", dtype=np.float)
        mass_out = pd.DataFrame(
            index=fluxes.index, columns=outcols, dtype=np.float)
        mass_in = fluxes.loc[:, incols].multiply(C_series)

        for t in fluxes.index:
            M_in = mass_in.loc[t].sum()

            M_out = V_out.loc[t] * C_out
            mass_out.loc[t] = M_out

            M = M + M_in + M_out.sum()

            mass_tot.loc[t] = M
            C_out = M / self.water.storage.loc[t, "storage"]

        self.logger.info("Simulation water quality succesfully completed.")
        return mass_in, mass_out, mass_tot

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
        }

        fluxes = self.water.fluxes.reindex(columns=d.keys())
        parsed_cols = fluxes.dropna(how="all", axis=1).columns.tolist()
        fluxes = fluxes.rename(columns=d)

        # Verhard: q_oa van alle Verhard bakjes
        names = ["q_oa_" + str(idn) for idn in self.buckets.keys() if
                 self.buckets[idn].name == "Verhard"]
        q_verhard = self.water.fluxes.loc[:, names]
        fluxes["verhard"] = q_verhard.sum(axis=1)

        # Uitspoeling: alle positieve q_ui fluxes uit alle verhard en onverhard en drain
        names = ["q_ui_" + str(idn) for idn in self.buckets.keys() if
                 self.buckets[idn].name in ["Verhard", "Onverhard"]]
        q_uitspoel = self.water.fluxes.loc[:, names]
        q_uitspoel[q_uitspoel < 0] = 0
        fluxes["uitspoeling"] = q_uitspoel.sum(axis=1)

        # Intrek: alle negatieve q_ui fluxes uit alle bakjes behalve MengRiool
        names = ["q_ui_" + str(idn) for idn in self.buckets.keys() if
                 self.buckets[idn].name != "MengRiool"]
        q_intrek = self.water.fluxes.loc[:, names]
        q_intrek[q_intrek > 0] = 0
        fluxes["intrek"] = q_intrek.sum(axis=1)

        # Oppervlakkige afstroming: q_oa van Onverharde en Drain bakjes
        names = ["q_oa_" + str(idn) for idn in self.buckets.keys() if
                 self.buckets[idn].name in ["Onverhard", "Drain"]]
        q_afstroom = self.water.fluxes.loc[:, names]
        fluxes["afstroming"] = q_afstroom.sum(axis=1)

        # Combined Sewer Overflow: q_cso van MengRiool bakjes
        names = ["q_cso_" + str(idn) for idn in self.buckets.keys() if
                 self.buckets[idn].name == "MengRiool"]
        q_cso = self.water.fluxes.loc[:, names]
        fluxes["q_cso"] = q_cso.sum(axis=1)

        # Gedraineerd: q_oa - positieve q_ui van Drain
        names = ["q_dr_" + str(idn) for idn in self.buckets.keys() if
                 self.buckets[idn].name == "Drain"]
        names2 = ["q_ui_" + str(idn) for idn in self.buckets.keys() if
                  self.buckets[idn].name == "Drain"]
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
            fluxes[icol.lower()] = self.water.fluxes[icol]

        return fluxes

    def aggregate_fluxes_w_pumpstation(self):
        fluxes = self.aggregate_fluxes()
        gemaal_cols = [
            icol for icol in self.series.columns if icol.lower().startswith("gemaal")]
        if len(gemaal_cols) == 0:
            self.logger.warning(
                "No timeseries for pumping station. Cannot aggregate.")
            return fluxes
        fluxes.rename(columns={"berekende uitlaat": "sluitfout"}, inplace=True)
        # Add pumping station timeseries to fluxes
        fluxes["maalstaat"] = -1*self.series.loc[:, gemaal_cols].sum(axis=1)
        # Calculate difference between calculated and measured pumped volume
        fluxes["sluitfout"] = fluxes["sluitfout"].subtract(fluxes["maalstaat"])
        # Correct inlet volume with difference between calculated and measured
        # fluxes["berekende inlaat"] = fluxes["berekende inlaat"] - fluxes.loc[fluxes.sluitfout<0, "sluitfout"]

        return fluxes

    def calculate_cumsum(self, fluxes_names=None, eagseries_names=None,
                         cumsum_period="year", month_offset=9, tmin=None,
                         tmax=None):
        # Get fluxes
        fluxes = self.aggregate_fluxes()

        if tmin is None:
            tmin = fluxes.index[0]
        if tmax is None:
            tmax = fluxes.index[-1]

        # Helper function to get grouper
        def get_grouper(s, cumsum_period="year", month_offset=9):
            if cumsum_period == "year":
                grouper = [(s.index - MonthOffset(n=month_offset)).year]
            elif cumsum_period == "month":
                grouper = [s.index.year, s.index.month]
            else:
                grouper = None
            return grouper

        # By default calculate cumsum for calculated in- and outflux
        if fluxes_names is None:
            fluxes_names = ["berekende uitlaat", "berekende inlaat"]

        fluxes = fluxes.loc[tmin:tmax, fluxes_names]
        # Get grouper for fluxes
        grouper = get_grouper(fluxes, cumsum_period=cumsum_period,
                              month_offset=month_offset)
        # Calculate cumsum fluxes
        if grouper is not None:
            cumsum_flux = fluxes.groupby(by=grouper).cumsum()
        else:
            cumsum_flux = fluxes.cumsum()

        # If eagseries_names defined calculate cumsum
        if eagseries_names is not None:
            # Get series based on column names
            series = self.series.loc[tmin:tmax, eagseries_names]
            # Get grouper for series
            grouper = get_grouper(series, cumsum_period=cumsum_period,
                                  month_offset=month_offset)
            # Calculate cumsum fluxes
            if grouper is not None:
                cumsum_series = series.groupby(by=grouper).cumsum()
            else:
                cumsum_series = series.cumsum()
            # Return both cumsum_flux and cumsum_series
            return cumsum_flux, cumsum_series
        # Return only cumsum_flux
        return cumsum_flux

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

    def calculate_missing_influx(self):
        """Calculate missing influx to the system by averaging the
        difference between the calculated outflux versus the measured
        outflux at the pumping station. The missing influx is equal to
        the average difference per month.

        Raises
        ------
        ValueError
            if no Pumping Station ("Gemaal") timeseries is in Eag.series
        AttributeError
            if the model has not yet been simulated

        Returns
        -------
        pd.Series
            Series containing the calcualted missing influx per day
        """
        gemaal_cols = [
            icol for icol in self.series.columns if icol.lower().startswith("gemaal")]
        if len(gemaal_cols) == 0:
            raise ValueError(
                "No series names starting with 'Gemaal' in eag.series!")
        if self.water.fluxes.empty:
            raise AttributeError("No simulation data! Simulate model!")
        fluxes = self.aggregate_fluxes()

        diff = self.series.loc[:, gemaal_cols].sum(
            axis=1) - -1*fluxes["berekende uitlaat"]
        diff.loc[diff <= 0.0] = 0.0

        inlaat_monthly = diff.resample("M").mean()
        inlaat_sluitfout = inlaat_monthly.resample("D").bfill()

        return inlaat_sluitfout

    def output_for_plots(self):

        output_dict = {}

        output_dict["{}_fluxes.csv".format(
            self.name)] = self.aggregate_fluxes()

        eagseries_names = None
        gemaal_cols = [
            icol for icol in self.series.columns if icol.lower().startswith("gemaal")]
        if len(gemaal_cols) > 0:
            eagseries_names = ["Gemaal"]
            output_dict["{}_fluxes_w_ps.csv".format(self.name)] = \
                self.aggregate_fluxes_w_pumpstation()

        cumsum = self.calculate_cumsum(eagseries_names=eagseries_names)
        if len(cumsum) == 2:
            for nam, iseries in zip(["inuitflux", "gemaal"], cumsum):
                output_dict["{}_{}_cumsum.csv".format(
                    self.name, nam)] = iseries

        output_dict["{}_fractions.csv".format(
            self.name)] = self.calculate_fractions()

        return output_dict

    def output_to_zipfile(self, zipfname, outputdict=None):

        import zipfile

        if outputdict is None:
            outputdict = self.output_for_plots()

        with zipfile.ZipFile(zipfname, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for csvname, series in outputdict.items():
                zipf.writestr(csvname, series.to_csv())
