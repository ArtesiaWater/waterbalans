"""This file contains practical classes and methods for use throughout the "Waterbalans" model.

"""
from pandas import to_datetime, to_timedelta, Series

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
              1.111, 1.429, 1.000]

    for i in range(1, 13):
        e[e.index.month == i] /= penman[i - 1]
    return e
