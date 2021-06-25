import os

import numpy as np
import pandas as pd
import pytest

from .test_002_run_eag import test_make_eag as make_eag


def setup_eag():
    e = make_eag()

    # add series
    N = 3650
    e.add_timeseries(pd.Series(index=pd.date_range("2000", periods=N, freq="D"),
                               data=1e-3 * np.random.rand(N)),
                     name="Neerslag", fillna=True, method=0.0)
    e.add_timeseries(pd.Series(index=pd.date_range("2000", periods=N, freq="D"),
                               data=0.25e-3 * np.random.rand(N)),
                     name="Verdamping", fillna=True, method=0.0)

    # load parameters
    test_data = r"./tests/data"
    params = pd.read_csv(os.path.join(test_data, "param_1396_3360-EAG-1.csv"),
                         delimiter=";", decimal=".")
    params["Waarde"] = pd.to_numeric(params.Waarde)

    return e, params


def setup_eag_wq():
    e, params = setup_eag()
    e.simulate(params)
    test_data = r"./tests/data"
    chloride_params = pd.read_csv(
        os.path.join(test_data, "stoffen_chloride_1396_3360-EAG-1.csv"),
        decimal=".", delimiter=";")
    chloride_params.replace("Riolering", "q_cso", inplace=True)
    return e, chloride_params


@pytest.mark.benchmark(group="simulate_eag")
def test_benchmark_simulate_numba(benchmark):
    e, params = setup_eag()
    _ = benchmark(e.simulate, params=params)
    return


@pytest.mark.benchmark(group="simulate_eag")
def test_benchmark_simulate_loop(benchmark):
    e, params = setup_eag()
    e.use_numba = False
    _ = benchmark(e.simulate, params=params)
    return


@pytest.mark.benchmark(group="simulate_eag_wq")
def test_benchmark_simulate_wq_numba(benchmark):
    e, wq_params = setup_eag_wq()
    _ = benchmark(e.simulate_wq, wq_params=wq_params)
    return


@pytest.mark.benchmark(group="simulate_eag_wq")
def test_benchmark_simulate_wq_loop(benchmark):
    e, wq_params = setup_eag_wq()
    e.use_numba = False
    _ = benchmark(e.simulate_wq, wq_params=wq_params)
    return
