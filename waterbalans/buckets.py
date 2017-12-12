"""This file contains the different classes for the buckets.



"""

from abc import ABC

import pandas as pd
from numba import jit

from waterbalans.io import load_series


class Bucket:
    __doc__ = """Class to construct a Bucket instance from a string. 

    """

    def __new__(cls, kind=None, *args, **kwargs):
        return eval(kind)(*args, **kwargs)


class BucketBase(ABC):
    __doc__ = """Base class from which all bucket classes inherit.
    
    """

    def __init__(self, polder=None, data=None, area=0.0):
        self.polder = polder  # Reference to mother object.
        self.data = data
        self.series = pd.DataFrame()
        # self.load_series()

        self.storage = pd.Series()
        self.fluxes = pd.DataFrame()

        self.parameters = pd.DataFrame(columns=["name", "initial", "optimal"])

        self.area = area  # area in square meters

    def load_series(self):
        series = dict()
        series["prec"] = self.polder.series["prec"]
        series["evap"] = self.polder.series["evap"]

        for name in ["seepage"]:
            series[name] = load_series(name)

        return series

    def calculate_wb(self, parameters, dt):
        pass

    def validate_wb(self):
        """Method to validate the water balance based on the total input,
        output and the change in storage of the model for each time step.

        Returns
        -------

        """
        pass


class Verhard(BucketBase):
    def __init__(self, polder, data, area=0.0):
        BucketBase.__init__(self, polder, data, area)

        self.series = pd.DataFrame(columns=["p", "e", "s"])
        self.fluxes = pd.DataFrame(columns=["q_no", "q_ui", "q_s", "q_oa"])
        self.storage = pd.DataFrame(columns=["boven", "onder"])

        self.parameters = pd.DataFrame(index=["v_eq", "v_max1", "v_max2",
                                              "fmin", "fmax", "i_fac", "u_fac",
                                              "n1", "n2"],
                                       columns=["name", "initial", "optimal"])

    def calculate_wb(self, parameters=None, dt=1.0):
        """Calculate the waterbalance for this bucket.

        Parameters
        ----------
        parameters
        dt

        Returns
        -------

        """

        if parameters is None:
            v_eq, v_max1, v_max2, fmin, fmax, i_fac, u_fac, n1, n2 \
                = self.parameters.loc[:, "optimal"]

        v1 = 0.0 * n1
        v2 = 0.5 * n2
        v_max1 = v_max1 * n1
        v_max2 = v_max2 * n2

        for t, pes in self.series.iterrows():
            p, e, s = pes
            q_no = calc_q_no(p, e, v1, v_eq, fmin, fmax, dt)
            q_ui = calc_q_ui(v2, i_fac, u_fac, v_eq, dt)
            q_s = calc_q_s(s, dt)
            v1, q_oa = vol_q_oa(v1, 0.0, q_no, 0.0, v_max1, dt)
            # The completely random choice to create a waterbalance rest term?
            if v1 < 0.0:
                v1 = 0.0
            v2, q_del = vol_q_oa(v2, q_s, 0.0, q_ui, v_max2, dt)
            self.storage.loc[t] = v1, v2
            self.fluxes.loc[
                t, ["q_no", "q_ui", "q_s", "q_oa"]] = q_no, q_ui, q_s, q_oa,


class Onverhard(BucketBase):
    def __init__(self, polder, data, area=0.0):
        BucketBase.__init__(self, polder, data, area)

        self.series = pd.DataFrame(columns=["p", "e", "s"])
        self.fluxes = pd.DataFrame(columns=["q_no", "q_ui", "q_s", "q_oa"])

        self.parameters = pd.DataFrame(index=["v_eq", "v_max", "fmin", "fmax",
                                              "i_fac", "u_fac", "n"],
                                       columns=["name", "initial", "optimal"])

    def calculate_wb(self, parameters=None, dt=1.0):
        """Calculate the waterbalance for this bucket.

        Parameters
        ----------
        parameters
        dt

        Returns
        -------

        """

        if parameters is None:
            v_eq, v_max, fmin, fmax, i_fac, u_fac, n = self.parameters.loc[:,
                                                       "optimal"]

        v = 0.2 * n
        v_max = v_max * n

        for t, pes in self.series.iterrows():
            p, e, s = pes
            q_no = calc_q_no(p, e, v, v_eq, fmin, fmax, dt)
            q_ui = calc_q_ui(v, i_fac, u_fac, v_eq, dt)
            q_s = calc_q_s(s, dt)
            v, q_oa = vol_q_oa(v, q_s, q_no, q_ui, v_max, dt)
            self.storage.loc[t] = v
            self.fluxes.loc[
                t, ["q_no", "q_ui", "q_s", "q_oa"]] = q_no, q_ui, q_s, q_oa,


class Drain(BucketBase):
    def __init__(self, polder, data, area=0.0):
        BucketBase.__init__(polder, data, area)
        self.series = pd.DataFrame(columns=["p", "e", "s"], dtype=float)
        self.fluxes = pd.DataFrame(columns=["q_no", "q_ui", "q_s", "q_oa",
                                            "q_dr"], dtype=float)
        self.storage = pd.DataFrame(columns=["boven", "onder"])

        self.parameters = pd.DataFrame(index=["v_eq", "v_max1", "v_max2",
                                              "fmin", "fmax", "i_fac",
                                              "u_fac1", "u_fac2",
                                              "n1", "n2"],
                                       columns=["name", "initial", "optimal"])

    @jit
    def calculate_wb(self, parameters=None, dt=1.0):
        """Calculate the waterbalance for this bucket.

        Parameters
        ----------
        parameters
        dt

        Returns
        -------

        """

        if parameters is None:
            v_eq, v_max1, v_max2, fmin, fmax, i_fac, u_fac1, u_fac2, n1, n2 \
                = self.parameters.loc[:, "optimal"]

        v1 = 0.35 * n1
        v2 = 0.15 * n2
        v_max1 = v_max1 * n1
        v_max2 = v_max2 * n2

        for t, pes in self.series.iterrows():
            p, e, s = pes
            q_no = calc_q_no(p, e, v1, v_eq, fmin, fmax, dt)
            q_boven = calc_q_ui(v1, i_fac, u_fac1, v_eq, dt)
            q_ui = calc_q_ui(v2, i_fac, u_fac2, v_eq, dt)
            q_s = calc_q_s(s, dt)
            v1, q_oa = vol_q_oa(v1, 0.0, q_no, q_boven, v_max1, dt)
            v2, q_dr = vol_q_oa(v2, q_s, -q_boven, q_ui, v_max2, dt)
            self.storage.loc[t] = v1, v2
            # self.fluxes.loc[
            #     t, ["q_no", "q_ui", "q_s", "q_oa", "q_dr"]] = q_no, q_ui, \
            #                                                   q_s, q_oa, q_dr
            self.fluxes.loc[t, ["q_no"]] = q_no
            self.fluxes.loc[t, ["q_ui"]] = q_ui
            self.fluxes.loc[t, ["q_s", ]] = q_s
            self.fluxes.loc[t, ["q_oa", ]] = q_oa
            self.fluxes.loc[t, ["q_dr"]] = q_dr


class Water(BucketBase):
    def __init__(self, polder, data, area=0.0):
        BucketBase.__init__(self, polder, data, area)
        self.series = pd.DataFrame(columns=["p", "e", "s"], dtype=float)

        self.fluxes = pd.DataFrame(columns=["q_no", "q_ui", "q_s", "q_oa",
                                            "q_dr"], dtype=float)
        self.storage = pd.DataFrame(columns=["boven", "onder"])

        self.parameters = pd.DataFrame(index=["v_eq", "v_max1", "v_max2",
                                              "fmin", "fmax", "i_fac",
                                              "u_fac1", "u_fac2",
                                              "n1", "n2"],
                                       columns=["name", "initial", "optimal"])

    def calculate_wb(self, parameters, dt=1.0):
        # 1. Add incoming fluxes from other buckets
        for bucket in self.polder.values():
            fluxes = bucket.fluxes.loc[:, ["q_ui", "q_oa"]]
            self.fluxes = self.fluxes.add(fluxes)  # Add values element wise?

        # 2. calculate water bucket specific fluxes
        p, e, s = self.series.loc[:, ["p", "e", "s"]]
        e = to_penman(e)
        self.fluxes.loc[:, "q_no"] = p - e
        self.fluxes.loc[:, "q_s"] = calc_q_s(s, dt)

        # for t, s in self.series.loc[:, ["s"]].iterrows():


@jit
def calc_q_no(p, e, v, v_eq, fmin, fmax, dt=1.0):
    """Method to calculate the precipitation excess

    Parameters
    ----------
    p: float
    e: float
    v: float
    v_eq: float
    fmin: float
    fmax: float
    dt: float

    Returns
    -------

    """
    if v < v_eq:
        q = (p - e * fmin) / dt
    else:
        q = (p - e * fmax) / dt

    return q


@jit
def calc_q_ui(v, i_fac, u_fac, v_eq, dt=1.0):
    if v < v_eq:
        q = (v * -i_fac) / dt
    else:
        q = (v * -u_fac) / dt
    return q



@jit
def calc_q_s(s, dt=1.0):
    return s / dt


@jit
def vol_q_oa(v, q_s, q_no, q_ui, v_max, dt=1.0):
    v_p = v + (q_s + q_ui + q_no) / dt
    if v_p > v_max:
        v = v_max
        q_oa = (v_max - v_p) / dt
    else:
        v = v_p
        q_oa = 0.0
    return v, q_oa


def to_penman(e):
    penman = [2.500, 1.071, 0.789, 0.769, 0.769, 0.763, 0.789, 0.838, 0.855,
              1.111, 1.429, 1.000]

    for i in range(1, 13):
        e[e.index.month == i] /= penman[i - 1]
    return e
