"""This file contains the base class for loading files. It primarily serves
as a wrapper around other load modules to allow dynamic selection of the
load module to use based on the file-format.

"""

import importlib
import os


def load_model(fname, id=None, ext="shp"):
    """Method to load a model from a file or database, with a dynamic import depending on the file extension.

    Returns
    -------
    data: dict
        dictionary describing the model structure.

    """
    if fname is None:
        fname = os.path.join("data", id + "." + ext)

    else:
        path, ext = os.path.splitext(fname)

    load_mod = importlib.import_module(ext, "waterbalans.io")
    data = load_mod.load(fname)

    return data


def load_series(fname=None, name=None, db_series="xml"):
    """Method to load time series from a file or database, with a dynamic
    import depending on the file extension.

    Returns
    -------

    """
    if fname is None:
        fname = os.path.join("data", "Tijdreeksen." + db_series)

    load_mod = importlib.import_module(db_series, "waterbalans.io")
    data = load_mod.load(fname)

    if type(name) == str:
        series = data[name]
    else:
        series = dict()
        for n in name:
            series[n] = data[name]

    return series


def load_parameters():
    """Method to load parameters from a file or database, with a dynamic
    import depending on the file extension.

    Returns
    -------

    """
    pass
