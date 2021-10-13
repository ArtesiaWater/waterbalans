import os

import numpy as np
import pandas as pd

import waterbalans as wb

test_data = r"./tests/data"


def test_make_gaf():
    name = "2110-GAF"
    gaf_id = 1557
    deelgebieden = pd.read_csv(os.path.join(test_data, "opp_1557_2110-GAF.csv"),
                               delimiter=";")
    g = wb.create_gaf(gaf_id, name, gafbuckets=deelgebieden)
    return g


def test_add_series_to_gaf():
    g = test_make_gaf()
    g.add_timeseries(pd.Series(index=pd.date_range("2000", periods=10, freq="D"),
                               data=1e-3*np.ones(10)), name="Neerslag", tmin="2000",
                     tmax="2000-01-10", fillna=True, method=0.0)
    g.add_timeseries(pd.Series(index=pd.date_range("2000", periods=10, freq="D"),
                               data=0.25e-3*np.ones(10)), name="Verdamping", tmin="2000",
                     tmax="2000-01-10", fillna=True, method=0.0)
    return g


def test_gaf_run():
    g = test_add_series_to_gaf()
    e, = g.get_eags()
    bm = e.get_buckets(buckettype="MengRiool")
    for b in bm:
        b.use_eag_cso_series = False
        b.path_to_cso_series = r"./tests/data/240_cso_timeseries.csv"

    # load parameters
    params = pd.read_csv(os.path.join(test_data, "param_1557_2110-GAF.csv"),
                         delimiter=";", decimal=",")
    # params.rename(columns={"ParamCode": "Code"}, inplace=True)
    params["Waarde"] = pd.to_numeric(params.Waarde)
    g.simulate(params, tmin="2000", tmax="2000-01-10")

    return g
