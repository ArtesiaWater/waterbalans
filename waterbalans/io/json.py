"""This file contains the json model file importer

"""
import json


def load(fname):
    """Method to load json files.

    Parameters
    ----------
    fname: str
        string containing the file path and name.

    Returns
    -------

    """
    with open(fname, "rb") as file:
        data = json.loads(file)

    return data
