import os

import numpy as np
import pandas as pd

import waterbalans as wb

test_data = r"./tests/data"


def test_make_eag():
    name = "3360-EAG-1"
    eag_id = 1396
    buckets = pd.read_csv(os.path.join(test_data, "opp_1396_3360-EAG-1.csv"), delimiter=";",
                          decimal=",")
    buckets["OppWaarde"] = pd.to_numeric(buckets.OppWaarde)
    e = wb.create_eag(eag_id, name, buckets, use_waterlevel_series=False)
    return e


def test_add_series_to_eag():
    e = test_make_eag()
    e.add_timeseries(pd.Series(index=pd.date_range("2000", periods=10, freq="D"), data=1e-3*np.ones(10)), name="Neerslag", tmin="2000", tmax="2000-01-10",
                     fillna=True, method=0.0)
    e.add_timeseries(pd.Series(index=pd.date_range("2000", periods=10, freq="D"), data=0.25e-3*np.ones(10)), name="Verdamping", tmin="2000", tmax="2000-01-10",
                     fillna=True, method=0.0)
    return e


def test_eag_run():
    e = test_add_series_to_eag()
    # load parameters
    params = pd.read_csv(os.path.join(test_data, "param_1396_3360-EAG-1.csv"), delimiter=";",
                         decimal=",")
    # params.rename(columns={"ParamCode": "Code"}, inplace=True)
    params["Waarde"] = pd.to_numeric(params.Waarde)
    # simulate
    e.simulate(params=params, tmin="2000", tmax="2000-01-10")
    e.water.validate()
    return e


def test_calculate_fluxes():
    e = test_eag_run()
    fluxes = e.aggregate_fluxes()
    return


def test_calculate_fractions():
    e = test_eag_run()
    fractions = e.calculate_fractions()
    return


def test_calculate_chloride():
    e = test_eag_run()
    chloride_params = pd.read_csv(os.path.join(test_data, "stoffen_chloride_1396_3360-EAG-1.csv"), 
                                  decimal=".", delimiter=";")
    chloride_params.columns = [icol.capitalize() for icol in chloride_params.columns]
    chloride_params.replace("Riolering", "q_cso", inplace=True)
    m = e.simulate_wq(chloride_params)
    return e, m


def test_add_real_series_and_simulate():
    tmin = "2000"
    tmax = "2001"
    # Lees de tijdreeksen in
    reeksen = pd.read_csv(os.path.join(test_data, "reeks_1396_3360-EAG-1.csv"), delimiter=";",
                          decimal=",")
    # get EAG
    e = test_make_eag()

    # add default series
    e.add_series(reeksen, tmin=tmin, tmax=tmax)

    # add external series
    series = pd.read_csv(os.path.join(
        test_data, "series_1396_3360-EAG-1.csv"), sep=";", parse_dates=True, index_col=[1])

    # Loop over Gemaal, Inlaat, Uitlaat:
    factor = 1.0
    for seriestype in ["Gemaal", "Inlaat", "Uitlaat"]:
        colmask = [True if icol.startswith(
            seriestype) else False for icol in series.columns]
        if np.sum(colmask) == 0:
            continue
        selected_series = series.loc[:, colmask]
        if seriestype == "Gemaal":
            selected_series = selected_series.sum(axis=1)
            e.add_timeseries(selected_series, name=seriestype, tmin=tmin, tmax=tmax,
                             fillna=True, method=0.0)
        else:
            if seriestype == "Uitlaat":
                factor = -1.0
            for jcol in selected_series.columns:
                e.add_timeseries(factor*selected_series.loc[:, jcol], name=jcol.split("|")[0],
                                 tmin=tmin, tmax=tmax, fillna=True, method=0.0)

    # Peil
    colmask = [True if icol.lower().startswith(
        "peil") else False for icol in series.columns]
    peil = series.loc[:, colmask]
    e.add_timeseries(peil, name="Peil", tmin=tmin, tmax=tmax,
                     fillna=True, method="ffill")

    # Neerslag en Verdamping
    intersection = series.index.intersection(e.series.index)
    colmask = [True if icol.lower().startswith(
        "neerslag") else False for icol in series.columns]
    e.series.loc[intersection, "Neerslag"] = series.loc[intersection,
                                                        colmask].fillna(0.0).values.squeeze() * 1e-3
    colmask = [True if icol.lower().startswith("verdamping")
               else False for icol in series.columns]
    e.series.loc[intersection, "Verdamping"] = series.loc[intersection,
                                                          colmask].fillna(0.0).values.squeeze() * 1e-3

    # load parameters
    params = pd.read_csv(os.path.join(test_data, "param_1396_3360-EAG-1.csv"), delimiter=";",
                         decimal=",")
    # params.rename(columns={"ParamCode": "Code"}, inplace=True)
    params["Waarde"] = pd.to_numeric(params.Waarde)
    # simulate
    e.simulate(params=params, tmin="2000", tmax="2005")

    return e
