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

    def __init__(self, id=None, eag=None, series=None, area=0.0):
        self.id = id
        self.eag = eag  # Reference to mother object.

        self.series = pd.DataFrame()
        self.series = self.series.append(series)
        self.load_series_from_eag()

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

        self.series["p"] = self.eag.series["p"]
        self.series["e"] = self.eag.series["e"]


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
    def __init__(self, id, eag, series, area=0.0):
        BucketBase.__init__(self, id, eag, series, area)
        self.name = "Verhard"

        self.parameters = pd.DataFrame(
            index=['VMax_1', 'VMax_2', 'VInit_1', 'VInit_2', 'EFacMin_1',
                   'EFacMax_1', 'RFacIn_2', 'RFacOut_2', 'por_1', 'por_2'],
            columns=["pname", "pinit", "popt", "pmin", "pmax", "pvary"])
        self.parameters.loc[:, "pname"] = self.parameters.index

        # Add bucket to the eag
        self.eag.add_bucket(self)

    def simulate(self, params, tmin=None, tmax=None, dt=1.0):
        """Calculate the waterbalance for this bucket.

        Parameters
        ----------
        params
        dt

        Returns
        -------

        """
        self.initialize(tmin=tmin, tmax=tmax)

        # Get parameters
        VMax_1, VMax_2, VInit_1, VInit_2, EFacMin_1, EFacMax_1, RFacIn_2, \
        RFacOut_2, por_1, por_2 = params.loc[self.parameters.index]

        v_eq = 0.0

        VMax_1 = VMax_1 * por_1
        VMax_2 = VMax_2 * por_2

        v1 = [VInit_1 * por_1]
        v2 = [VInit_2 * por_2]
        q_no = []
        q_ui = []
        q_s = []
        q_oa = []

        for t, pes in self.series.iterrows():
            p, e, s = pes
            q_no.append(
                calc_q_no(p, e, v1[-1], v_eq, EFacMin_1, EFacMax_1, dt))
            q_ui.append(calc_q_ui(v2[-1], RFacIn_2, RFacOut_2, v_eq, dt))
            q_s.append(s)
            v, q = vol_q_oa(v1[-1], 0.0, q_no[-1], 0.0, VMax_1, dt)
            # The completely random choice to create a waterbalance rest term?
            v1.append(max(0.0, v))
            q_oa.append(q)
            v, q_del = vol_q_oa(v2[-1], q_s[-1], 0.0, q_ui[-1], VMax_2, dt)
            v2.append(v)

        self.fluxes = self.fluxes.assign(q_no=q_no, q_ui=q_ui, q_s=q_s,
                                         q_oa=q_oa)

        self.storage = self.storage.assign(Upper_Storage=v1[1:],
                                           Lower_Storage=v2[1:])


class Onverhard(BucketBase):
    def __init__(self, id, eag, series, area=0.0):
        BucketBase.__init__(self, id, eag, series, area)
        self.name = "Onverhard"

        self.parameters = pd.DataFrame(
            index=['VMax_1', 'VInit_1', 'EFacMin_1', 'EFacMax_1', 'RFacIn_1',
                   'RFacOut_1', 'por_1'],
            columns=["pname", "pinit", "popt", "pmin", "pmax", "pvary"])
        self.parameters.loc[:, "pname"] = self.parameters.index

        # Add bucket to the eag
        self.eag.add_bucket(self)

    def simulate(self, params, tmin=None, tmax=None, dt=1.0):
        """Calculate the waterbalance for this bucket.

        Parameters
        ----------
        params
        dt

        Returns
        -------

        """
        self.initialize(tmin=tmin, tmax=tmax)

        # Get parameters
        VMax_1, VInit_1, EFacMin_1, EFacMax_1, RFacIn_1, RFacOut_1, por_1 = \
            params.loc[self.parameters.index]

        VMax_1 = VMax_1 * por_1

        v_eq = 0.0

        v = [VInit_1 * por_1]
        q_no = []
        q_ui = []
        q_s = []
        q_oa = []

        for t, pes in self.series.iterrows():
            p, e, s = pes
            q_no.append(calc_q_no(p, e, v[-1], v_eq, EFacMin_1, EFacMax_1, dt))
            q_ui.append(calc_q_ui(v[-1], RFacIn_1, RFacOut_1, v_eq, dt))
            q_s.append(s)
            v1, q = vol_q_oa(v[-1], q_s[-1], q_no[-1], q_ui[-1], VMax_1, dt)
            v.append(v1)
            q_oa.append(q)

        self.fluxes = self.fluxes.assign(q_no=q_no, q_ui=q_ui, q_s=q_s,
                                         q_oa=q_oa)
        self.storage = self.storage.assign(Storage=v[1:])


class Drain(BucketBase):
    def __init__(self, id, eag, series, area=0.0):
        BucketBase.__init__(self, id, eag, series, area)
        self.name = "Drain"

        self.parameters = pd.DataFrame(
            index=['VMax_1', 'VMax_2', 'VInit_1', 'VInit_2', 'EFacMin_1',
                   'EFacMax_1', 'RFacIn_2', 'RFacOut_1', 'RFacOut_2', 'por_1',
                   'por_2'],
            columns=["pname", "pinit", "popt", "pmin", "pmax", "pvary"])
        self.parameters.loc[:, "pname"] = self.parameters.index

        # Add bucket to the eag
        self.eag.add_bucket(self)

    def simulate(self, params, tmin=None, tmax=None, dt=1.0):
        """Calculate the waterbalance for this bucket.

        Parameters
        ----------
        params
        dt

        Returns
        -------

        """
        self.initialize(tmin=tmin, tmax=tmax)

        # Get parameters
        VMax_1, VMax_2, VInit_1, VInit_2, EFacMin_1, EFacMax_1, RFacIn_2, \
        RFacOut_1, RFacOut_2, por_1, por_2 = \
            params.loc[self.parameters.index]

        v_eq = 0.0

        VMax_1 = VMax_1 * por_1
        VMax_2 = VMax_2 * por_2

        v1 = [VInit_1 * por_1]
        v2 = [VInit_2 * por_2]
        q_no = []
        q_ui = []
        q_s = []
        q_oa = []
        q_dr = []

        for t, pes in self.series.iterrows():
            p, e, s = pes
            q_no.append(
                calc_q_no(p, e, v1[-1], v_eq, EFacMin_1, EFacMax_1, dt))
            q_boven = calc_q_ui(v1[-1], RFacIn_2, RFacOut_2, v_eq, dt)
            q_ui.append(calc_q_ui(v2[-1], RFacIn_2, RFacOut_2, v_eq, dt))
            q_s.append(s)
            v, q = vol_q_oa(v1[-1], 0.0, q_no[-1], q_boven, VMax_1, dt)
            v1.append(v)
            q_oa.append(q)
            v, q = vol_q_oa(v2[-1], q_s[-1], -q_boven, q_ui[-1], VMax_2, dt)
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
