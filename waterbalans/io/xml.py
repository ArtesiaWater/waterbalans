"""This file contains the XML-importer for the waterbalance module.

"""
import pandas as pd
import xmltodict

from waterbalans.io.decorators import filepath_or_buffer


@filepath_or_buffer
def read_xml(filepath_or_buffer):
    file = filepath_or_buffer

    data = xmltodict.parse(file)

    series = data["TimeSeries"]["series"]
    header = series["header"]
    series = pd.DataFrame(series["event"])
    index = pd.to_datetime(series["@date"] + " " + series["@time"])
    series.set_index(index, inplace=True)
    series.drop(["@time", "@date", "@flag"], axis=1, inplace=True)
    series = series.squeeze().astype(float)
    series.name = header['locationId']

    series.metadata = header

    return series


def to_xml(data, fname):
    #with open(fname, "wb"):
    raise NotImplementedError

