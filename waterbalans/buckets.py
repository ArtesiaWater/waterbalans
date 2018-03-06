"""This file contains the different classes for the buckets.



"""

from abc import ABC

import pandas as pd


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

        self.parameters = pd.DataFrame(columns=["bucket", "pname", "pinit",
                                                "popt", "pmin", "pmax",
                                                "pvary"])
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
            pass
            # series[name] = load_series(name)

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
                                              "v_init1", "v_init2", "fmin",
                                              "fmax", "i_fac", "u_fac",
                                              "n1", "n2"],
                                       columns=["pname", "pinit", "popt",
                                                "pmin", "pmax", "pvary"])
        self.parameters.loc[:, "pname"] = self.parameters.index

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
            parameters = self.parameters.loc[:, "popt"]

        v_eq, v_max1, v_max2, v_init1, v_init2, fmin, fmax, i_fac, u_fac, n1, \
        n2 = parameters

        v_max1 = v_max1 * n1
        v_max2 = v_max2 * n2

        v1 = [v_init1 * n1]
        v2 = [v_init2 * n2]
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

        self.parameters = pd.DataFrame(index=["v_eq", "v_max", "v_init",
                                              "fmin", "fmax", "i_fac",
                                              "u_fac", "n"],
                                       columns=["pname", "pinit", "popt",
                                                "pmin", "pmax", "pvary"])
        self.parameters.loc[:, "pname"] = self.parameters.index

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
            parameters = self.parameters.loc[:, "popt"]

        v_eq, v_max, v_init, fmin, fmax, i_fac, u_fac, n = parameters

        v_max = v_max * n

        v = [v_init * n]
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
                                              "v_init1", "v_init2",
                                              "fmin", "fmax", "i_fac",
                                              "u_fac1", "u_fac2", "n1", "n2"],
                                       columns=["pname", "pinit", "popt",
                                                "pmin", "pmax", "pvary"])
        self.parameters.loc[:, "pname"] = self.parameters.index

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
            parameters = self.parameters.loc[:, "popt"]

        v_eq, v_max1, v_max2, v_init1, v_init2, fmin, fmax, i_fac, u_fac1, \
        u_fac2, n1, n2 = parameters

        v_max1 = v_max1 * n1
        v_max2 = v_max2 * n2

        v1 = [v_init1 * n1]
        v2 = [v_init2 * n2]
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
