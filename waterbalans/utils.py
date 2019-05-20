"""This file contains practical classes and methods for use throughout the "Waterbalans" model.

"""
import os

import numpy as np
import pandas as pd
from pandas import to_datetime, to_timedelta


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args,
                                                                 **kwargs)
        return cls._instances[cls]


def excel2datetime(excel_datenum, freq="D", start_date="1899-12-30"):
    """Method to convert excel datetime to pandas timetime objects.

    Parameters
    ----------
    excel_datenum: datetime index
        can be a datetime object or a pandas datetime index.
    freq:

    Returns
    -------
    datetimes: pandas.datetimeindex

    """
    datetimes = to_datetime(start_date) + to_timedelta(excel_datenum, freq)
    return datetimes


def makkink_to_penman(e, use_excel_factors=False):
    """Method to transform the the makkink potential evaporation to Penman
    evaporation for open water.

    Parameters
    ----------
    e: pandas.Series
        Pandas Series containing the evaporation with the date as index.
    use_excel_factors: bool, optional default is False
        if True, uses the excel factors, only difference is that evaporation
        in december is 0.0.

    Returns
    -------
    e: pandas.Series
        Penman evaporation as a pandas time series object.

    Notes
    -----
    Van Penman naar Makkink, een nieuwe berekeningswijze voor de
    klimatologische verdampingsgetallen, KNMI/CHO, rapporten en nota's, no.19

    """
    if use_excel_factors:
        penman = [2.500, 1.071, 0.789, 0.769, 0.769, 0.763, 0.789, 0.838, 0.855,
                  1.111, 1.429, np.inf]  # col E47:E59 in Excel e_r / e_o, with 0 evap in december.
    else:
        penman = [2.500, 1.071, 0.789, 0.769, 0.769, 0.763, 0.789, 0.838, 0.855,
                  1.111, 1.429, 1.000]  # col E47:E59 in Excel e_r / e_o
    e = e.copy()
    for i in range(1, 13):
        e.loc[e.index.month == i] = e.loc[e.index.month == i] / \
            penman[i - 1]  # for first list
    return e


def calculate_cso(prec, Bmax, POCmax, alphasmooth=0.1):
    """Calculate Combined Sewer Overflow timeseries based
    on hourly precipitation series.

    Parameters
    ----------
    prec : pd.Series
        hourly precipitation
    Bmax : float
        maximum storage capacity in meters
    POCmax : float
        maximum pump (over) capacity
    alphasmooth : float, optional
        factor for exponential smoothing (the default is 0.1)

    Returns
    -------
    pd.Series
        timeseries of combined sewer overflows (cso)

    """

    p_smooth = prec.ewm(alpha=alphasmooth, adjust=False).mean()
    b = p_smooth.copy()
    poc = p_smooth.copy()
    cso = p_smooth.copy()
    vol = p_smooth.copy()

    for i in range(1, len(p_smooth.index)):
        vol.iloc[i] = p_smooth.iloc[i] + b.iloc[i-1] - poc.iloc[i-1]
        b.iloc[i] = np.min([vol.iloc[i], Bmax])
        poc.iloc[i] = np.min([b.iloc[i], POCmax])
        cso.iloc[i] = np.max([vol.iloc[i] - Bmax, 0.0])

    cso_daily = cso.resample("D").sum()

    return cso_daily


def get_model_input_from_excel(excelfile):
    """get modelstructure, timeseries, and parameters from an excel file.
    The structure of the excelfile is defined. See example file at
    https://github.com/ArtesiaWater/waterbalans/tree/master/voorbeelden/data

    Parameters
    ----------
    excelfile : str, path to excel file
        Path to excelfile

    Returns
    -------
    df_ms, df_ts, df_params: pandas.DataFrames
        DataFrames containing info about modelstructure,
        timeseries and parameters.

    """
    xls = pd.ExcelFile(excelfile)

    df_ms = pd.read_excel(xls, sheet_name="modelstructure", skiprows=[1],
                          header=[0], index_col=None)
    df_ts = pd.read_excel(xls, sheet_name="reeksen", skiprows=[1],
                          header=[0], index_col=None, usecols="A:J")
    df_params = pd.read_excel(xls, sheet_name="parameters", skiprows=[1],
                              header=[0], index_col=None, usecols="A:G")

    return df_ms, df_ts, df_params


def get_extra_series_from_excel(excelfile, sheet_name="extra_reeksen"):
    """Load extra timeseries from excelfile containing all info
    for an EAG. The structure of the excelfile is defined. See example file at
    https://github.com/ArtesiaWater/waterbalans/tree/master/voorbeelden/data

    Parameters
    ----------
    excelfile : str, path to excelfile
        path to excelfile

    Returns
    -------
    df_series: pandas.DataFrame
        DataFrame containing series to be added to waterbalance

    """

    xls = pd.ExcelFile(excelfile)
    df_series = pd.read_excel(xls, sheet_name=sheet_name, skiprows=[1],
                              header=[0], index_col=[0], parse_dates=True)

    return df_series


def get_wqparams_from_excel(excelfile, sheet_name="stoffen"):
    """Load water quality parameters from excelfile containing all info
    for an EAG. The structure of the excelfile is defined. See example file at
    https://github.com/ArtesiaWater/waterbalans/tree/master/voorbeelden/data

    Parameters
    ----------
    excelfile : str, path to excelfile
        path to excelfile

    Returns
    -------
    df_series: pandas.DataFrame
        DataFrame containing water quality parameters

    """

    xls = pd.ExcelFile(excelfile)
    df_series = pd.read_excel(xls, sheet_name=sheet_name, skiprows=[1],
                              header=[0], index_col=None, usecols="A:I")

    return df_series


def get_extra_series_from_pickle(picklefile, compression="zip"):
    """Load timeseries from pickle.

    Parameters
    ----------
    picklefile : str, path to picklefile
        path to picklefile

    Returns
    -------
    df_series: pandas.DataFrame
        DataFrame containing series

    """

    df_series = pd.read_pickle(picklefile, compression=compression)
    return df_series


def add_timeseries_to_obj(eag_or_gaf, df, tmin=None, tmax=None, overwrite=False,
                          data_from_excel=False):
    """Add timeseries to EAG or GAF. Only parses column names starting with
    'Neerslag', 'Verdamping', 'Inlaat', 'Uitlaat', 'Peil', or 'Gemaal'.

    Parameters
    ----------
    eag_or_gaf : waterbalans.Eag or waterbalans.Gaf
        An existing Eag or Gaf object
    df : pandas.DataFrame
        DataFrame with DateTimeIndex containing series to be added
    tmin : pandas.TimeStamp, optional
        start time for added series (the default is None, which
        attempts to pick up tmin from existing Eag or Gaf object)
    tmax : pandas.TimeStamp, optional
        end time for added series (the default is None, which
        attempts to pick up tmax from existing Eag or Gaf object)
    overwrite : bool, optional
        overwrite series if name already exists in Eag or Gaf object (the default is False)
    data_from_excel: bool, optional
        if True, assumes data source is Excel Balance 'uitgangspunten' sheet.
        Function will make an assumption about the column names and order and
        disregards column names of the passed DataFrame. If False uses DataFrame
        column names (default).

    """
    o = eag_or_gaf
    # if not isinstance(o, Eag) or not isinstance(o, Gaf):
    #     raise ValueError(
    #         "Arg 'eag_or_gaf' must be Eag or Gaf object. Received {}".format(type(o)))

    try:
        if tmin is None:
            tmin = o.series.index[0]
        if tmax is None:
            tmax = o.series.index[-1]
    except IndexError:
        raise ValueError(
            "tmin/tmax cannot be inferred from EAG/GAF object.")

    if data_from_excel:
        columns = ["neerslag", "verdamping", "peil",
                   "Gemaal1", "Gemaal2", "Gemaal3", "Gemaal4",
                   "Inlaat voor calibratie", "gemengd gerioleerd stelsel",
                   "Inlaat1", "Inlaat2", "Inlaat3", "Inlaat4",
                   "Uitlaat1", "Uitlaat2", "Uitlaat3", "Uitlaat4"]
    else:
        columns = df.columns

    eag_series = o.series.columns

    # Inlaat/Uitlaat
    factor = 1.0
    for inam in ["Gemaal", "Inlaat", "Uitlaat"]:
        colmask = [True if icol.lower().startswith(inam.lower())
                   else False for icol in columns]
        series = df.loc[:, colmask]
        # Water bucket converts outgoing fluxes to negative, so outgoing fluxes can be entered positive
        for jcol in range(series.shape[1]):
            # Check if empty
            if series.iloc[:, jcol].dropna().empty:
                o.logger.warning("'{}' is empty. Continuing...".format(
                    series.columns[jcol]))
                continue
            # Check if series is already in EAG
            if data_from_excel:
                colnam = np.array(columns)[colmask][jcol]
            else:
                colnam = series.columns[jcol].split("|")[0]
            if colnam in eag_series:
                if overwrite:
                    o.add_timeseries(factor*series.iloc[:, jcol], name="{}".format(colnam),
                                     tmin=tmin, tmax=tmax, fillna=True, method=0.0)
                else:
                    o.logger.warning("'{}' already in EAG. No action taken.".format(
                        colnam))
            else:
                # o.logger.info("Adding '{}' series to EAG.".format(
                #     colnam))
                o.add_timeseries(factor*series.iloc[:, jcol], name="{}".format(colnam),
                                 tmin=tmin, tmax=tmax, fillna=True, method=0.0)

    # Peil
    colmask = [True if icol.lower().startswith("peil")
               else False for icol in columns]
    if np.sum(colmask) > 0:
        peil = df.loc[:, colmask]
        if "Peil" in eag_series:
            if overwrite:
                o.add_timeseries(peil, name="Peil", tmin=tmin, tmax=tmax,
                                 fillna=True, method="ffill")
            else:
                o.logger.warning("'Peil' already in EAG. No action taken.")
        else:
            # o.logger.info("Adding 'Peil' series to EAG.")
            o.add_timeseries(peil, name="Peil", tmin=tmin, tmax=tmax,
                             fillna=True, method="ffill")

    # q_cso MengRiool overstortreeks
    colmask = [True if icol.lower().startswith("q_cso")
               else False for icol in columns]
    if np.sum(colmask) > 0:
        q_cso = df.loc[:, colmask] / 100**2
        if "q_cso" in eag_series:
            if overwrite:
                o.add_timeseries(q_cso, name="q_cso", tmin=tmin, tmax=tmax,
                                 fillna=True, method=0.0)
            else:
                o.logger.warning("'q_cso' already in EAG. No action taken.")
        else:
            # o.logger.info("Adding 'q_cso' series to EAG.")
            o.add_timeseries(q_cso, name="q_cso", tmin=tmin, tmax=tmax,
                             fillna=True, method=0.0)

    # Neerslag/Verdamping
    for inam in ["Neerslag", "Verdamping"]:
        colmask = [True if icol.lower().startswith(inam.lower())
                   else False for icol in columns]
        if np.sum(colmask) > 0:
            pe = df.loc[:, colmask] * 1e-3
            if inam in eag_series:
                if overwrite:
                    o.add_timeseries(pe, name=inam, tmin=tmin, tmax=tmax,
                                     fillna=True, method=0.0)
                else:
                    o.logger.warning(
                        "'{}' already in EAG. No action taken.".format(inam))
            else:
                # o.logger.info("Adding '{}' series to EAG.".format(inam))
                o.add_timeseries(pe, name=inam, tmin=tmin, tmax=tmax,
                                 fillna=True, method=0.0)


def create_csvfile_table(csvdir):
    """Creates a DataFrame containing all csv file
    names for EAGs or GAFS in that folder.

    Parameters
    ----------
    csvdir : path to dir
        folder in which csvs are stored

    Returns
    -------
    pandas.DataFrame
        DataFrame containing each CSV for a specific EAG or GAF
    """

    files = [i for i in os.listdir(csvdir) if i.endswith(".csv")]
    eag_df = pd.DataFrame(data=files, columns=["filenames"])
    eag_df["ID"] = eag_df.filenames.apply(
        lambda s: s.split("_")[2].split(".")[0])
    eag_df["type"] = eag_df.filenames.apply(lambda s: s.split(
        "_")[0] if not s.startswith("stoffen") else "_".join(s.split("_")[:2]))
    eag_df.drop_duplicates(subset=["ID", "type"], keep="last", inplace=True)
    file_df = eag_df.pivot(index="ID", columns="type", values="filenames")
    file_df.dropna(how="any", subset=[
        "opp", "param", "reeks"], axis=0, inplace=True)
    return file_df


def compare_to_excel_balance(e, pickle_dir="./data/excel_pklz", **kwargs):
    # Read Excel Balance Data (see scrape_excelbalansen.py for details)
    excelbalance = pd.read_pickle(os.path.join(pickle_dir, "{}_wbalance.pklz".format(e.name)),
                                  compression="zip")
    for icol in excelbalance.columns:
        excelbalance.loc[:, icol] = pd.to_numeric(
            excelbalance[icol], errors="coerce")

    # Waterbalance comparison
    fig = e.plot.compare_fluxes_to_excel_balance(
        excelbalance, **kwargs)

    return fig
