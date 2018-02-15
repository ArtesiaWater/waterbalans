"""This file contains the different classes for the buckets.



"""

from abc import ABC

import pandas as pd

from waterbalans.io import load_series


class Bucket:
    __doc__ = """Class to construct a Bucket instance from a string. 

    """

    def __new__(cls, kind=None, *args, **kwargs):
        return eval(kind)(*args, **kwargs)


class BucketBase(ABC):
    __doc__ = """Base class from which all bucket classes inherit.
    
    """

    def __init__(self, eag=None, series=None, area=0.0):
        self.eag = eag  # Reference to mother object.
        self.series = pd.DataFrame()

        if series is None:
            self.load_series_from_eag()
        else:
            self.series = series

        self.parameters = pd.DataFrame(columns=["name", "initial", "optimal"])

        self.area = area  # area in square meters

    def initialize(self, tmin=None, tmax=None):
        """Method to initialize a Bucket with a clean DataFrame for the
        fluxes and storage time series. This method is called by the init
        and simulate methods.

        """
        if tmin is None:
            tmin = self.series.index.min()
        else:
            tmin = pd.Timestamp(tmin)

        if tmax is None:
            tmax = self.series.index.max()
        else:
            tmax = pd.Timestamp(tmax)

        index = self.series[tmin:tmax].index
        self.fluxes = pd.DataFrame(index=index, dtype=float)
        self.storage = pd.DataFrame(index=index, dtype=float)

    def load_series_from_eag(self):
        series = dict()
        series["prec"] = self.eag.series["prec"]
        series["evap"] = self.eag.series["evap"]

        for name in ["seepage"]:
            series[name] = load_series(name)

        return series

    def simulate(self, parameters, tmin=None, tmax=None, dt=1.0):
        pass

    def validate(self):
        """Method to validate the water balance based on the total input,
        output and the change in storage of the model for each time step.

        Returns
        -------

        """
        pass


class Verhard(BucketBase):
    def __init__(self, eag, series, area=0.0):
        BucketBase.__init__(self, eag, series, area)
        self.name = "Verhard"
        self.parameters = pd.DataFrame(index=["v_eq", "v_max1", "v_max2",
                                              "fmin", "fmax", "i_fac", "u_fac",
                                              "n1", "n2"],
                                       columns=["name", "initial", "optimal"])

    def simulate(self, parameters=None, tmin=None, tmax=None, dt=1.0):
        """Calculate the waterbalance for this bucket.

        Parameters
        ----------
        parameters
        dt

        Returns
        -------

        """
        self.initialize(tmin=tmin, tmax=tmax)

        if parameters is None:
            parameters = self.parameters.loc[:, "optimal"]

        v_eq, v_max1, v_max2, fmin, fmax, i_fac, u_fac, n1, n2 = parameters

        v_max1 = v_max1 * n1
        v_max2 = v_max2 * n2

        v1 = [0.0 * n1]
        v2 = [0.5 * n2]
        q_no = []
        q_ui = []
        q_s = []
        q_oa = []

        for t, pes in self.series.iterrows():
            p, e, s = pes
            q_no.append(calc_q_no(p, e, v1[-1], v_eq, fmin, fmax, dt))
            q_ui.append(calc_q_ui(v2[-1], i_fac, u_fac, v_eq, dt))
            q_s.append(s)
            v, q = vol_q_oa(v1[-1], 0.0, q_no[-1], 0.0, v_max1, dt)
            # The completely random choice to create a waterbalance rest term?
            if v < 0.0:
                v = 0.0
            v1.append(v)
            q_oa.append(q)
            v, q_del = vol_q_oa(v2[-1], q_s[-1], 0.0, q_ui[-1], v_max2, dt)
            v2.append(v)

        self.fluxes = self.fluxes.assign(q_no=q_no, q_ui=q_ui, q_s=q_s,
                                         q_oa=q_oa)

        self.storage = self.storage.assign(Upper_Storage=v1[1:],
                                           Lower_Storage=v2[1:])


class Onverhard(BucketBase):
    def __init__(self, eag, series, area=0.0):
        BucketBase.__init__(self, eag, series, area)
        self.name = "Onverhard"

        self.parameters = pd.DataFrame(index=["v_eq", "v_max", "fmin", "fmax",
                                              "i_fac", "u_fac", "n"],
                                       columns=["name", "initial", "optimal"])

    def simulate(self, parameters=None, tmin=None, tmax=None, dt=1.0):
        """Calculate the waterbalance for this bucket.

        Parameters
        ----------
        parameters
        dt

        Returns
        -------

        """
        self.initialize(tmin=tmin, tmax=tmax)

        if parameters is None:
            parameters = self.parameters.loc[:, "optimal"]

        v_eq, v_max, fmin, fmax, i_fac, u_fac, n = parameters

        v_max = v_max * n

        v = [0.2 * n]
        q_no = []
        q_ui = []
        q_s = []
        q_oa = []

        for t, pes in self.series.iterrows():
            p, e, s = pes
            q_no.append(calc_q_no(p, e, v[-1], v_eq, fmin, fmax, dt))
            q_ui.append(calc_q_ui(v[-1], i_fac, u_fac, v_eq, dt))
            q_s.append(s)
            v1, q = vol_q_oa(v[-1], q_s[-1], q_no[-1], q_ui[-1], v_max, dt)
            v.append(v1)
            q_oa.append(q)

        self.fluxes = self.fluxes.assign(q_no=q_no, q_ui=q_ui, q_s=q_s,
                                         q_oa=q_oa)
        self.storage = self.storage.assign(Storage=v[1:])


class Drain(BucketBase):
    def __init__(self, eag, series, area=0.0):
        BucketBase.__init__(self, eag, series, area)
        self.name = "Drain"

        self.parameters = pd.DataFrame(index=["v_eq", "v_max1", "v_max2",
                                              "fmin", "fmax", "i_fac",
                                              "u_fac1", "u_fac2", "n1", "n2"],
                                       columns=["name", "initial", "optimal"])

    def simulate(self, parameters=None, tmin=None, tmax=None, dt=1.0):
        """Calculate the waterbalance for this bucket.

        Parameters
        ----------
        parameters
        dt

        Returns
        -------

        """
        self.initialize(tmin=tmin, tmax=tmax)

        if parameters is None:
            parameters = self.parameters.loc[:, "optimal"]

        v_eq, v_max1, v_max2, fmin, fmax, i_fac, u_fac1, u_fac2, n1, n2 = parameters

        v_max1 = v_max1 * n1
        v_max2 = v_max2 * n2

        v1 = [0.35 * n1]
        v2 = [0.15 * n2]
        q_no = []
        q_ui = []
        q_s = []
        q_oa = []
        q_dr = []

        for t, pes in self.series.iterrows():
            p, e, s = pes
            q_no.append(calc_q_no(p, e, v1[-1], v_eq, fmin, fmax, dt))
            q_boven = calc_q_ui(v1[-1], i_fac, u_fac1, v_eq, dt)
            q_ui.append(calc_q_ui(v2[-1], i_fac, u_fac2, v_eq, dt))
            q_s.append(s)
            v, q = vol_q_oa(v1[-1], 0.0, q_no[-1], q_boven, v_max1, dt)
            v1.append(v)
            q_oa.append(q)
            v, q = vol_q_oa(v2[-1], q_s[-1], -q_boven, q_ui[-1], v_max2, dt)
            v2.append(v)
            q_dr.append(q)

        self.fluxes = self.fluxes.assign(q_no=q_no, q_ui=q_ui, q_s=q_s,
                                         q_oa=q_oa, q_dr=q_dr)

        self.storage = self.storage.assign(Upper_Storage=v1[1:],
                                           Lower_Storage=v2[1:])


class Water(BucketBase):
    def __init__(self, eag, series, area=0.0):
        BucketBase.__init__(self, eag, series, area)

        self.name = "Water"
        self.parameters = pd.DataFrame(
            index=["h_eq", "h_min", "h_max", "q_max"],
            columns=["name", "initial", "optimal"])

    def simulate(self, parameters=None, tmin=None, tmax=None, dt=1.0):
        self.initialize(tmin=tmin, tmax=tmax)

        if parameters is None:
            parameters = self.parameters.loc[:, "optimal"]

        h_eq, h_min, h_max, q_max = parameters

        # 1. Add incoming fluxes from other buckets
        for bucket in self.eag.buckets.values():
            if bucket.name == "Water":
                pass
            else:
                names = ["q_ui", "q_oa", "q_dr"]
                names = [name for name in names if
                         name in bucket.fluxes.columns]
                fluxes = bucket.fluxes.loc[:, names] * -bucket.area
                fluxes.columns = [name + "_" + bucket.name for name in names]
                self.fluxes = self.fluxes.join(fluxes, how="outer")

        # 2. calculate water bucket specific fluxes
        series = self.series.multiply(self.area)
        self.fluxes.loc[:, "q_p"] = series.loc[:, "p"]
        self.fluxes.loc[:, "q_e"] = -makkink_to_penman(series.loc[:, "e"])
        self.fluxes.loc[:, "q_s"] = series.loc[:, "s"]
        self.fluxes.loc[:, "q_w"] = -series.loc[:, "w"]

        h_min = h_min * self.area
        h_max = h_max * self.area

        h = [h_eq * self.area]
        q_in = []
        q_out = []

        q_totals = self.fluxes.sum(axis=1)

        # 3. Calculate the fluxes coming in and going out.
        for q_tot in q_totals.values:
            # Calculate the outgoing flux
            if h[-1] + q_tot > h_max:
                q_out.append(min(q_max, h_max - h[-1] - q_tot))
                q_in.append(0.0)
            elif h[-1] + q_tot < h_min:
                q_in.append(h_min - h[-1] - q_tot)
                q_out.append(0.0)
            else:
                q_out.append(0.0)
                q_in.append(0.0)

            h.append(h[-1] + q_in[-1] + q_out[-1] + q_tot)

        self.storage = pd.Series(data=h[1:], index=self.fluxes.index)
        self.fluxes = self.fluxes.assign(q_in=q_in, q_out=q_out)


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


def calc_q_ui(v, i_fac, u_fac, v_eq, dt=1.0):
    if v < v_eq:
        q = (v * -i_fac) / dt
    else:
        q = (v * -u_fac) / dt
    return q


def vol_q_oa(v, q_s, q_no, q_ui, v_max, dt=1.0):
    """Method to calculate the storage and the q_oa flux.

    Parameters
    ----------
    v: float
        storage at previous time step (t-1)
    q_s: float
        seepage flux
    q_no: float
        precipitation excess flux
    q_ui: float
        ... flux
    v_max: float
        maximam storage volume
    dt: float
        timestep

    Returns
    -------
    v: float
        storage at current time step.
    q_oa: float
        outgoing flux

    """
    v_p = v + (q_s + q_ui + q_no) / dt
    if v_p > v_max:
        v = v_max
        q_oa = (v_max - v_p) / dt
    else:
        v = v_p
        q_oa = 0.0
    return v, q_oa


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
