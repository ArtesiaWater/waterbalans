"""This file contains practical classes and methods for use throughout the "Waterbalans" model.

"""
import os
import numpy as np
import pandas as pd
from pandas import to_datetime, to_timedelta, Series
import matplotlib.pyplot as plt
from pastas.read import KnmiStation

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

def makkink_to_penman(e):
    """Method to transform the the makkink potential evaporation to Penman
    evaporation for open water.

    Parameters
    ----------
    e: pandas.Series
        Pandas Series containing the evaporation with the date as index.

    Returns
    -------
    e: pandas.Series
        Penman evaporation as a pandas time series object.

    Notes
    -----
    Van Penman naar Makkink, een nieuwe berekeningswijze voor de
    klimatologische verdampingsgetallen, KNMI/CHO, rapporten en nota's, no.19

    """
    penman = [2.500, 1.071, 0.789, 0.769, 0.769, 0.763, 0.789, 0.838, 0.855,
              1.111, 1.429, 1.000]  # col E47:E59 in Excel e_r / e_o
    # penman = [2.500, 1.071, 0.789, 0.769, 0.769, 0.763, 0.789, 0.838, 0.855,
    #         1.111, 1.429, np.inf]  # col E47:E59 in Excel e_r / e_o, with 0 evap in december.
    # TODO: which one to use? 2019/02/14 --> this second list seems odd, checking 
    # with maker of excel balance which to use. Probably bug in Excel!
    # penman = [0.400, 0.933, 1.267, 1.300, 1.300, 1.310, 1.267, 1.193, 1.170, 
    #           0.900, 0.700, 0.000]  # col D47:D59 in Excel e_o / e_r

    e = e.copy()
    for i in range(1, 13):
        e.loc[e.index.month == i] = e.loc[e.index.month == i] / penman[i - 1]  # for first list
        # e.loc[e.index.month == i] = e.loc[e.index.month == i] * penman[i - 1]  # for second list
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
