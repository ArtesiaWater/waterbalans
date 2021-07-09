import logging
import os

import numpy as np
import pandas as pd
import waterbalans as wb


# %% data dir and test data
test_data = r"./tests/data"

index = pd.date_range("2019-01-01", "2019-01-31", freq="D")
neerslag = pd.Series(index=index, data=np.random.random_sample(31)) * 1e-3
neerslag.name = "Neerslag"
verdamping = 0.75 * \
    pd.Series(index=index, data=np.random.random_sample(31)) * 1e-3
verdamping.name = "Verdamping"
series = pd.concat([neerslag, verdamping], axis=1)


def test_simulate_bucket_verhard():
    e = wb.Eag(idn=1, name="test_eag", log_level=logging.DEBUG)
    b1 = wb.buckets.Verhard(10, e, series=series, area=100.0)

    # numba
    b1.simulate(b1.parameters)
    f1, s1 = b1.fluxes, b1.storage

    # loop
    e.use_numba = False
    b1.simulate(b1.parameters)
    f2, s2 = b1.fluxes, b1.storage

    # check
    assert np.allclose(f1, f2)
    assert np.allclose(s1, s2)
    return


def test_simulate_bucket_onverhard():
    e = wb.Eag(idn=1, name="test_eag", log_level=logging.DEBUG)
    b2 = wb.buckets.Onverhard(20, e, series=series, area=100.0)

    # numba
    b2.simulate(b2.parameters)
    f1, s1 = b2.fluxes, b2.storage

    # loop
    e.use_numba = False
    b2.simulate(b2.parameters)
    f2, s2 = b2.fluxes, b2.storage

    # check
    assert np.allclose(f1, f2)
    assert np.allclose(s1, s2)
    return


def test_simulate_bucket_drain():
    e = wb.Eag(idn=1, name="test_eag", log_level=logging.DEBUG)
    b3 = wb.buckets.Drain(30, e, series=series, area=100.0)

    # numba
    b3.simulate(b3.parameters)
    f1, s1 = b3.fluxes, b3.storage

    # loop
    e.use_numba = False
    b3.simulate(b3.parameters)
    f2, s2 = b3.fluxes, b3.storage

    # check
    assert np.allclose(f1, f2)
    assert np.allclose(s1, s2)
    return


# def test_bucket_mengriool():
#     e = wb.Eag(idn=1, name="test_eag", log_level=logging.DEBUG)
#     b4 = wb.buckets.MengRiool(40, e, series=series, area=100.0)
#     b4.simulate(b4.parameters)
#     return


def test_simulate_bucket_water():
    e = wb.Eag(idn=1, name="test_eag", log_level=logging.DEBUG)
    w = wb.water.Water(50, e, series=series, area=100.0)
    # numba
    w.simulate(w.parameters)
    f1, s1 = w.fluxes, w.storage

    # loop
    e.use_numba = False
    w.simulate(w.parameters)
    f2, s2 = w.fluxes, w.storage

    # check
    assert np.allclose(f1, f2)
    assert np.allclose(s1, s2)
    return


def test_make_eag():
    name = "3360-EAG-1"
    eag_id = 1396
    buckets = pd.read_csv(os.path.join(test_data, "opp_1396_3360-EAG-1.csv"),
                          delimiter=";", decimal=".")
    buckets["OppWaarde"] = pd.to_numeric(buckets.OppWaarde)
    e = wb.create_eag(eag_id, name, buckets, use_waterlevel_series=False,
                      log_level=logging.DEBUG)
    return e


def test_add_series_to_eag():
    e = test_make_eag()
    e.add_timeseries(pd.Series(index=pd.date_range("2000", periods=10, freq="D"), data=1e-3 * np.ones(10)), name="Neerslag", tmin="2000", tmax="2000-01-10",
                     fillna=True, method=0.0)
    e.add_timeseries(pd.Series(index=pd.date_range("2000", periods=10, freq="D"), data=0.25e-3 * np.ones(10)), name="Verdamping", tmin="2000", tmax="2000-01-10",
                     fillna=True, method=0.0)
    return e


def test_eag_run():
    e = test_add_series_to_eag()
    # load parameters
    params = pd.read_csv(os.path.join(test_data, "param_1396_3360-EAG-1.csv"),
                         delimiter=";", decimal=".")
    # params.rename(columns={"ParamCode": "Code"}, inplace=True)
    params["Waarde"] = pd.to_numeric(params.Waarde)
    # simulate
    e.simulate(params=params, tmin="2000", tmax="2000-01-10")
    e.water.validate()
    return e


def test_calculate_fluxes():
    e = test_eag_run()
    _ = e.aggregate_fluxes()
    return


def test_calculate_fractions():
    e = test_eag_run()
    _ = e.calculate_fractions()
    return


def test_calculate_chloride():
    e = test_eag_run()
    chloride_params = pd.read_csv(os.path.join(test_data, "stoffen_chloride_1396_3360-EAG-1.csv"),
                                  decimal=".", delimiter=";")
    # chloride_params.columns = [icol.capitalize()
    #                            for icol in chloride_params.columns]
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
    e.add_series_from_database(reeksen, tmin=tmin, tmax=tmax)

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
                e.add_timeseries(factor * selected_series.loc[:, jcol], name=jcol.split("|")[0],
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


def test_compare_numba_simulate():
    # load parameters
    params = pd.read_csv(os.path.join(test_data, "param_1396_3360-EAG-1.csv"),
                         delimiter=";", decimal=".")
    # params.rename(columns={"ParamCode": "Code"}, inplace=True)
    params["Waarde"] = pd.to_numeric(params.Waarde)
    # simulate

    e1 = test_add_series_to_eag()
    e1.simulate(params=params, tmin="2000", tmax="2000-01-10")

    e2 = test_add_series_to_eag()
    e2.use_numba = False
    e2.simulate(params=params, tmin="2000", tmax="2000-01-10")

    assert np.allclose(e1.aggregate_fluxes().dropna(how="any"),
                       e2.aggregate_fluxes().dropna(how="any"))

    return


def test_compare_numba_simulate_wq():
    chloride_params = pd.read_csv(
        os.path.join(test_data, "stoffen_chloride_1396_3360-EAG-1.csv"),
        decimal=".", delimiter=";")
    chloride_params.replace("Riolering", "q_cso", inplace=True)

    e1 = test_eag_run()
    m1 = e1.simulate_wq(chloride_params)

    e2 = test_eag_run()
    e2.use_numba = False
    m2 = e2.simulate_wq(chloride_params)

    assert np.allclose(m1[-1], m2[-1])

    return


def test_compare_numba_fractions():
    e1 = test_eag_run()
    f1 = e1.calculate_fractions()

    e2 = test_eag_run()
    e2.use_numba = False
    f2 = e2.calculate_fractions().loc[:, f1.columns].astype(float)

    assert np.allclose(f1, f2)
    return


def test_compare_numba_fractions_large_inflow():
    e1 = test_eag_run()
    # add large inflow
    inlaat = pd.Series(index=e1.series.index, data=1e6)
    e1.add_timeseries(inlaat, "Inlaat1")

    # load parameters
    params = pd.read_csv(os.path.join(test_data, "param_1396_3360-EAG-1.csv"),
                         delimiter=";", decimal=".")
    # params.rename(columns={"ParamCode": "Code"}, inplace=True)
    params["Waarde"] = pd.to_numeric(params.Waarde)
    # set storage limits
    params.loc[19, "Waarde"] = -0.05
    params.loc[20, "Waarde"] = -0.05

    # simulate
    e1.simulate(params=params, tmin="2000", tmax="2000-01-10")
    f1 = e1.calculate_fractions()

    e2 = test_eag_run()
    # add large inflow
    inlaat = pd.Series(index=e2.series.index, data=1e6)
    e2.add_timeseries(inlaat, "Inlaat1")
    e2.use_numba = False
    e2.simulate(params=params, tmin="2000", tmax="2000-01-10")
    f2 = e2.calculate_fractions().loc[:, f1.columns].astype(float)

    assert np.allclose(f1, f2)
    return
