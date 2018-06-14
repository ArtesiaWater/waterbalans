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
    
    Parameters
    ----------
    id: int, optional
        Integer id of the bucket. This id is also used to connect parameters.
    eag: waterbalans.Eag, optional
        Eag instance where this bucket is appended to the Eag.buckets dict.
    
    """

    def __init__(self, id=None, eag=None, series=None, area=0.0):
        self.id = id
        self.eag = eag  # Reference to mother object.
        # Add bucket to the eag
        self.eag.add_bucket(self)

        self.series = pd.DataFrame()
        self.series = self.series.append(series)
        self.load_series_from_eag()

        self.parameters = pd.DataFrame(columns=["Waarde"])
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
        """Method to automatically load precipitation and evaporation from
        for the eag if available.

        """
        if self.eag is None:
            return

        if "prec" in self.eag.series.columns:
            self.series["prec"] = self.eag.series["prec"]
        if "evap" in self.eag.series.columns:
            self.series["evap"] = self.eag.series["evap"]

    def simulate(self, parameters, tmin=None, tmax=None, dt=1.0):
        """Calculate the waterbalance for this bucket.

        Parameters
        ----------
        params: pandas.Series
            Series with the parameters. If not all parameters are provided,
            the default parameters are used for simulation.
        tmin: str or pandas.Timestamp
        tmax: str or pandas.Timestamp
        dt: float, optional
            float value with the time step used for simulation. Not used
            right now.

        """
        pass


class Verhard(BucketBase):
    def __init__(self, id, eag, series, area=0.0):
        BucketBase.__init__(self, id, eag, series, area)
        self.name = "Verhard"

        self.parameters = pd.DataFrame(
            data=[0.002, 1, 0.5, 1, 1, 0.1, 0.1, 0.2],
            index=['hMax_1', 'hMax_2', 'hInit_1', 'EFacMin_1',
                   'EFacMax_1', 'RFacIn_2', 'RFacOut_2', 'por_2'],
            columns=["Waarde"])

    def simulate(self, params, tmin=None, tmax=None, dt=1.0):
        self.initialize(tmin=tmin, tmax=tmax)
        # Get parameters
        self.parameters.update(params)
        hMax_1, hMax_2, hInit_2, EFacMin_1, EFacMax_1, RFacIn_2, RFacOut_2, \
        por_2 = self.parameters.loc[:, "Waarde"]

        hEq = 0.0

        hMax_2 = hMax_2 * por_2

        h_1 = [0]  # Initial storage is zero
        h_2 = [hInit_2 * por_2]  # initial storage is 0.5 times the porosity
        q_no = []
        q_ui = []
        q_s = []
        q_oa = []

        for t, pes in self.series.loc[:, ["prec", "evap", "seep"]].iterrows():
            p, e, s = pes
            q_no.append(
                calc_q_no(p, e, h_1[-1], hEq, EFacMin_1, EFacMax_1, dt))
            q_ui.append(calc_q_ui(h_2[-1], RFacIn_2, RFacOut_2, hEq, dt))
            q_s.append(s)
            v, q = vol_q_oa(h_1[-1], 0.0, q_no[-1], 0.0, hMax_1, dt)
            # The completely random choice to create a waterbalance rest term?
            h_1.append(max(0.0, v))
            q_oa.append(q)
            v, q_del = vol_q_oa(h_2[-1], q_s[-1], 0.0, q_ui[-1], hMax_2, dt)
            h_2.append(v)

        self.fluxes = self.fluxes.assign(q_no=q_no, q_ui=q_ui, q_s=q_s,
                                         q_oa=q_oa)

        self.storage = self.storage.assign(Upper_Storage=h_1[1:],
                                           Lower_Storage=h_2[1:])


class Onverhard(BucketBase):
    def __init__(self, id, eag, series, area=0.0):
        BucketBase.__init__(self, id, eag, series, area)
        self.name = "Onverhard"

        self.parameters = pd.DataFrame(
            data=[0.5, 0.5, 0.75, 1.0, 0.01, 0.02, 0.1],
            index=['hMax_1', 'hInit_1', 'EFacMin_1', 'EFacMax_1',
                   'RFacIn_1', 'RFacOut_1', 'por_1'],
            columns=["Waarde"])


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
        self.parameters.update(params)
        hMax_1, hInit_1, EFacMin_1, EFacMax_1, RFacIn_1, RFacOut_1, por_1 = \
            self.parameters.loc[:, "Waarde"]

        hMax_1 = hMax_1 * por_1

        hEq = 0.0

        h = [hInit_1 * por_1]
        q_no = []
        q_ui = []
        q_s = []
        q_oa = []

        for t, pes in self.series.loc[:, ["prec", "evap", "seep"]].iterrows():
            p, e, s = pes
            q_no.append(calc_q_no(p, e, h[-1], hEq, EFacMin_1, EFacMax_1, dt))
            q_ui.append(calc_q_ui(h[-1], RFacIn_1, RFacOut_1, hEq, dt))
            q_s.append(s)
            v1, q = vol_q_oa(h[-1], q_s[-1], q_no[-1], q_ui[-1], hMax_1, dt)
            h.append(v1)
            q_oa.append(q)

        self.fluxes = self.fluxes.assign(q_no=q_no, q_ui=q_ui, q_s=q_s,
                                         q_oa=q_oa)
        self.storage = self.storage.assign(Storage=h[1:])


class Drain(BucketBase):
    def __init__(self, id, eag, series, area=0.0):
        BucketBase.__init__(self, id, eag, series, area)
        self.name = "Drain"

        self.parameters = pd.DataFrame(
            data=[],  # TODO Vul de waarden in
            index=['VMax_1', 'VMax_2', 'VInit_1', 'VInit_2', 'EFacMin_1',
                   'EFacMax_1', 'RFacIn_2', 'RFacOut_1', 'RFacOut_2', 'por_1',
                   'por_2'],
            columns=["Waarde"])
        self.parameters.loc[:, "pname"] = self.parameters.index


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

        for t, pes in self.series.loc[:, ["prec", "evap", "seep"]].iterrows():
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


def calc_q_no(p, e, h, hEq, EFacMin, EFacMax, dt=1.0):
    """Method to calculate the precipitation excess.

    Parameters
    ----------
    p: float
        Precipitation.
    e: float
        Evaporation.
    h: float
        Waterlevel.
    hEq: float
        Waterlevel equilibrium.
    EFacMin: float
        Minimum evaporation factor.
    EFacMax: float
        Maximum evaporation factor.
    dt: float
        Timestep, not used right now.

    Returns
    -------
    q: float
        Precipitation excess.

    """
    if h < hEq:
        q = (p - e * EFacMin) / dt
    else:
        q = (p - e * EFacMax) / dt
    return q


def calc_q_ui(h, RFacIn, RFacOut, hEq, dt=1.0):
    """Method to calculate the lateral in- and outflow of the bucket.

    Parameters
    ----------
    h: float
        Waterlevel in the bucket.
    RFacIn: float
        Factor to determine the incoming flux.
    RFacOut:float
        Factor to determine the outgoing flux.
    hEq: float
        Equilibrium waterlevel.
    dt: float
        Timestep for used in the calculation. Not used right now.

    Returns
    -------
    q_ui: float
        inflow (positive) or outflow (negative) flux.

    Notes
    -----
    When the waterlevel is above the equilibrium level, the returned flux is
    the waterlevel times a factor for outgoing fluxes, otherwise the
    waterlevel (h) times the factor for the ingoing fluxea.

    """
    if h < hEq:
        q_ui = (h * -RFacIn) / dt
    else:
        q_ui = (h * -RFacOut) / dt
    return q_ui


def vol_q_oa(v, q_s, q_no, q_ui, hMax, dt=1.0):
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
    hMax: float
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
    if v_p > hMax:
        v = hMax
        q_oa = (hMax - v_p) / dt
    else:
        v = v_p
        q_oa = 0.0
    return v, q_oa
