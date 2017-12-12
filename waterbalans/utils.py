"""This file contains practical classes and methods for use throughout the "Waterbalans" model.

"""
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
