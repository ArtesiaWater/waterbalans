"""This file contains the different classes for the buckets.



"""
from abc import ABC

import pandas as pd

from .utils import calculate_cso


class Bucket:
    __doc__ = """Class to construct a Bucket instance from a string.

    """

    def __new__(cls, kind=None, *args, **kwargs):
        return eval(kind)(*args, **kwargs)


class BucketBase(ABC):
    __doc__ = """Base class from which all bucket classes inherit.

    Parameters
    ----------
    idn: int, optional
        Integer ID of the bucket. This ID is also used to connect parameters.
    eag: waterbalans.Eag, optional
        Eag instance where this bucket is appended to the Eag.buckets dict.

    """

    def __init__(self, idn=None, eag=None, series=None, area=0.0):
        self.idn = idn
        self.eag = eag  # Reference to mother object.

        # Add bucket to the eag
        self.eag.add_bucket(self)

        self.series = pd.DataFrame()
        self.series = self.series.append(series)

        self.parameters = pd.DataFrame(columns=["Waarde"])
        self.area = area  # area in square meters

    def initialize(self, tmin=None, tmax=None):
        """Method to initialize a Bucket with a clean DataFrame for the
        fluxes and storage time series. This method is called by the init
        and simulate methods.

        """
        self.load_series_from_eag()

        if tmin is None:
            tmin = self.series.index.min()
        else:
            tmin = pd.Timestamp(tmin)

        if tmax is None:
            tmax = self.series.index.max()
        else:
            tmax = pd.Timestamp(tmax)

        index = self.series[tmin:tmax].dropna().index

        self.fluxes = pd.DataFrame(index=index, dtype=float)
        self.storage = pd.DataFrame(index=index, dtype=float)

    def load_series_from_eag(self):
        """Method to automatically load Precipitation and Evaporation from
        for the eag if available.

        """
        if self.eag is None:
            return

        if "Neerslag" in self.eag.series.columns:
            self.series["Neerslag"] = self.eag.series["Neerslag"]
        if "Verdamping" in self.eag.series.columns:
            self.series["Verdamping"] = self.eag.series["Verdamping"]

    def simulate(self, params=None, tmin=None, tmax=None, dt=1.0):
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

    def __repr__(self):
        return "<{0}: {1} bucket with area {2:.1f}>".format(self.idn,
                                                            self.name,
                                                            self.area)


class Verhard(BucketBase):
    def __init__(self, idn, eag, series, area=0.0):
        BucketBase.__init__(self, idn, eag, series, area)
        self.name = "Verhard"

        self.parameters = pd.DataFrame(
            data=[0.002, 1, 0.5, 1, 1, 0.1, 0.1, 0.2],
            index=['hMax_1', 'hMax_2', 'hInit_2', 'EFacMin_1',
                   'EFacMax_1', 'RFacIn_2', 'RFacOut_2', 'por_2'],
            columns=["Waarde"])

    def simulate(self, params=None, tmin=None, tmax=None, dt=1.0):
        self.initialize(tmin=tmin, tmax=tmax)

        # Get parameters
        msg = "{0} {1}: using default parameter value {2} for '{3}'"
        for ipar in self.parameters.index.difference(params.index):
            self.eag.logger.debug(msg.format(
                self.name, self.idn, self.parameters.loc[ipar, "Waarde"], ipar))

        self.parameters.update(params)

        hMax_1, hMax_2, hInit_2, EFacMin_1, EFacMax_1, RFacIn_2, RFacOut_2, \
            por_2 = self.parameters.loc[:, "Waarde"]

        hEq = 0.0

        hMax_2 = hMax_2 * por_2

        h_1 = [0]  # Initial storage is zero
        h_2 = [hInit_2 * por_2]
        q_no = []
        q_ui = []
        q_s = []
        q_oa = []

        series = self.series.loc[self.fluxes.index]

        # test if columns are present!
        if not {"Neerslag", "Verdamping", "Qkwel"}.issubset(series.columns):
            msg = "Warning: {} not in series. Assumed equal to 0!"
            self.eag.logger.warning(msg.format(
                {"Neerslag", "Verdamping", "Qkwel"} - set(series.columns)))

        for _, pes in series.reindex(columns=["Neerslag", "Verdamping", "Qkwel"],
                                     fill_value=0.0).iterrows():
            p, e, s = pes

            # Bereken de waterbalans in laag 1
            q_no.append(
                calc_q_no(p, e, h_1[-1], hEq, EFacMin_1, EFacMax_1, dt))
            h, q = calc_h_q_oa(h_1[-1], 0.0, q_no[-1], 0.0, hMax_1, dt)

            # Interception reservoir storage cannot be negative
            h_1.append(max(0.0, h))
            q_oa.append(q)

            # Bereken de waterbalans in laag 2
            q_s.append(s)
            q_ui.append(calc_q_ui(h_2[-1], RFacIn_2, RFacOut_2, hEq, dt))
            h, _ = calc_h_q_oa(h_2[-1], s, 0.0, q_ui[-1], hMax_2, dt)
            h_2.append(h)

        self.fluxes = self.fluxes.assign(q_no=q_no, q_ui=q_ui, q_s=q_s,
                                         q_oa=q_oa)

        self.storage = self.storage.assign(Upper_Storage=h_1[1:],
                                           Lower_Storage=h_2[1:])


class Onverhard(BucketBase):
    def __init__(self, idn, eag, series, area=0.0):
        BucketBase.__init__(self, idn, eag, series, area)
        self.name = "Onverhard"

        self.parameters = pd.DataFrame(
            data=[0.5, 0.5, 0.75, 1.0, 0.01, 0.02, 0.1],
            index=['hMax_1', 'hInit_1', 'EFacMin_1', 'EFacMax_1',
                   'RFacIn_1', 'RFacOut_1', 'por_1'],
            columns=["Waarde"])

    def simulate(self, params=None, tmin=None, tmax=None, dt=1.0):
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
        msg = "{0} {1}: using default parameter value {2} for '{3}'"
        for ipar in self.parameters.index.difference(params.index):
            self.eag.logger.debug(msg.format(
                self.name, self.idn, self.parameters.loc[ipar, "Waarde"], ipar))

        self.parameters.update(params)
        hMax_1, hInit_1, EFacMin_1, EFacMax_1, RFacIn_1, RFacOut_1, por_1 = \
            self.parameters.loc[:, "Waarde"]

        hMax_1 = hMax_1 * por_1

        hEq = 0.0

        h_1 = [hInit_1 * por_1]
        q_no = []
        q_ui = []
        q_s = []
        q_oa = []

        series = self.series.loc[self.fluxes.index]

        # test if columns are present!
        if not {"Neerslag", "Verdamping", "Qkwel"}.issubset(series.columns):
            self.eag.logger.warning("Warning: {} not in series. Assumed equal to 0!".format(
                {"Neerslag", "Verdamping", "Qkwel"} - set(series.columns)))
        for _, pes in series.reindex(columns=["Neerslag", "Verdamping", "Qkwel"],
                                     fill_value=0.0).iterrows():
            p, e, s = pes
            q_no.append(
                calc_q_no(p, e, h_1[-1], hEq, EFacMin_1, EFacMax_1, dt))
            qui = calc_q_ui(h_1[-1], RFacIn_1, RFacOut_1, hEq, dt)
            q_ui.append(qui)
            q_s.append(s)
            h, q = calc_h_q_oa(h_1[-1], s, q_no[-1], q_ui[-1], hMax_1, dt)
            h_1.append(h)
            q_oa.append(q)

        self.fluxes = self.fluxes.assign(q_no=q_no, q_ui=q_ui, q_s=q_s,
                                         q_oa=q_oa)
        self.storage = self.storage.assign(Storage=h_1[1:])


class Drain(BucketBase):
    def __init__(self, idn, eag, series, area=0.0):
        BucketBase.__init__(self, idn, eag, series, area)
        self.name = "Drain"

        self.parameters = pd.DataFrame(
            data=[0.7, 0.3, 0.35, 0.3, 0.75, 1.0,
                  0.5, 0.0, 0.001, 0.001, 0.3, 0.3],
            index=['hMax_1', 'hMax_2', 'hInit_1', 'hInit_2',
                   'EFacMin_1', 'EFacMax_1', 'RFacIn_1', 'RFacIn_2',
                   'RFacOut_1', 'RFacOut_2', 'por_1', 'por_2'],
            columns=["Waarde"])
        # self.parameters.loc[:, "pname"] = self.parameters.index

    def simulate(self, params=None, tmin=None, tmax=None, dt=1.0):
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
        non_defined_params = set(self.parameters.index) - set(params.index)
        if len(non_defined_params) > 0:
            msg = "Warning: {} not set in parameters, using default values!"
            self.eag.logger.warning(msg.format(non_defined_params))

        self.parameters.update(params)

        hMax_1, hMax_2, hInit_1, hInit_2, EFacMin_1, EFacMax_1, RFacIn_1, RFacIn_2, \
            RFacOut_1, RFacOut_2, por_1, por_2 = self.parameters.loc[:, "Waarde"]

        hEq = 0.0

        hMax_1 = hMax_1 * por_1
        hMax_2 = hMax_2 * por_2

        h_1 = [hInit_1 * por_1]
        h_2 = [hInit_2 * por_2]
        q_no = []
        q_ui = []
        q_s = []
        q_oa = []
        q_dr = []

        series = self.series.loc[self.fluxes.index]

        # test if columns are present!
        if not {"Neerslag", "Verdamping", "Qkwel"}.issubset(series.columns):
            msg = "Warning Bucket {0}-{1}: {2} not in series. Assumed equal to 0!"
            self.eag.logger.warning(msg.format(
                self.name, self.idn,
                {"Neerslag", "Verdamping", "Qkwel"} - set(series.columns)))

        for _, pes in series.reindex(columns=["Neerslag", "Verdamping", "Qkwel"],
                                     fill_value=0.0).iterrows():
            p, e, s = pes
            no = calc_q_no(p, e, h_1[-1], hEq, EFacMin_1, EFacMax_1, dt)
            q_no.append(no)
            q_boven = calc_q_ui(h_1[-1], RFacIn_1, RFacOut_1, hEq, dt)
            q_ui.append(calc_q_ui(h_2[-1], RFacIn_2, RFacOut_2, hEq, dt))
            q_s.append(s)
            h, q = calc_h_q_oa(h_1[-1], 0.0, q_no[-1], q_boven, hMax_1, dt)
            h_1.append(h)
            q_oa.append(q)
            h, q = calc_h_q_oa(h_2[-1], s, -q_boven, q_ui[-1], hMax_2, dt)
            h_2.append(h)
            q_dr.append(q)

        self.fluxes = self.fluxes.assign(q_no=q_no, q_ui=q_ui, q_s=q_s,
                                         q_oa=q_oa, q_dr=q_dr)

        self.storage = self.storage.assign(Upper_Storage=h_1[1:],
                                           Lower_Storage=h_2[1:])


class MengRiool(BucketBase):
    def __init__(self, idn, eag, series, area=0.0, use_eag_cso_series=True,
                 path_to_cso_series=None):
        BucketBase.__init__(self, idn, eag, series, area)
        self.name = "MengRiool"
        self.parameters = pd.DataFrame(
            data=[240, 5e-3, 0.5e-3],
            index=['KNMIStation', 'Bmax', 'POCmax'],
            columns=["Waarde"])
        self.use_eag_cso_series = use_eag_cso_series
        self.path_to_cso_series = path_to_cso_series

    def simulate(self, params=None, tmin=None, tmax=None, dt=1.0):
        self.initialize(tmin=tmin, tmax=tmax)

        # Get parameters
        msg = "{0} {1}: using default parameter value {2} for '{3}'"
        for ipar in self.parameters.index.difference(params.index):
            self.eag.logger.debug(msg.format(
                self.name, self.idn, self.parameters.loc[ipar, "Waarde"], ipar))
        self.parameters.update(params)

        knmistn = int(self.parameters.at["KNMIStation", "Waarde"])
        Bmax = self.parameters.at["Bmax", "Waarde"]
        POCmax = self.parameters.at["POCmax", "Waarde"]

        # See if cached version is available, otherwise calculate
        # Note caching has risks if Bmax and POCmax change!
        # And also if a different period is calculated!
        try:
            if self.use_eag_cso_series:
                ts_cso = self.eag.series.loc[pd.to_datetime(
                    tmin):pd.to_datetime(tmax), "q_cso"]
                self.eag.logger.info(
                    "Picked up CSO timeseries from EAG object.")
            else:
                fcso = self.path_to_cso_series
                if fcso is None:
                    raise TypeError(
                        "Set path to external cso timeseries file!")
                if fcso.endswith(".pklz"):
                    ts_cso = pd.read_pickle(fcso, compression="zip")
                elif fcso.endswith(".csv"):
                    ts_cso = pd.read_csv(fcso, index_col=[0], parse_dates=True,
                                         header=None)
                else:
                    raise NotImplementedError(
                        "External CSO timeseries file must have extension .pklz or .csv!")
                ts_cso = ts_cso.loc[pd.to_datetime(tmin):pd.to_datetime(tmax)]
                self.eag.logger.info(
                    "Picked up CSO timeseries from external file.")
        except (FileNotFoundError, KeyError) as e:
            try:
                from pastas.read import KnmiStation
            except ModuleNotFoundError as e:
                self.eag.logger.exception("Module 'pastas' not installed! Please intall using "
                                          "pip to automatically donwload KNMI data!")
                raise e
            self.eag.logger.error(
                "Failed loading CSO series from EAG or from external file.")
            self.eag.logger.warning(
                "Calculating CSO series... (this can take a while).")
            self.eag.logger.info(
                "Downloading hourly KNMI data for station {}".format(knmistn))
            prec = KnmiStation.download(
                stns=[knmistn], interval="hour", start=tmin, end=tmax, vars="RH")
            self.eag.logger.info(
                "KNMI Download succeeded, calculating series...")
            ts_cso = calculate_cso(prec.data.RH, Bmax, POCmax, alphasmooth=0.1)
            self.eag.logger.info("CSO series calculated.")

        series = pd.Series(index=ts_cso.index,
                           data=-1.*ts_cso.values.squeeze())
        series.name = "q_cso"

        self.fluxes = self.fluxes.assign(q_cso=series)
        self.storage = self.storage.assign(Storage=0.)


def calc_q_no(p, e, h, hEq, EFacMin, EFacMax, dt=1.0):
    """Method to calculate the Precipitation excess.

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
        Minimum Evaporation factor.
    EFacMax: float
        Maximum Evaporation factor.
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
    # TODO minus sign could/should be in parameter
    if h < hEq:
        q_ui = (h * -RFacIn) / dt
    else:
        q_ui = (h * -RFacOut) / dt
    return q_ui


def calc_h_q_oa(h, q_s, q_no, q_ui, hMax, dt=1.0):
    """Method to calculate the storage h and the flux q_oa.

    Parameters
    ----------
    h: float
        storage at previous time step (t-1)
    q_s: float
        seepage flux
    q_no: float
        Precipitation excess flux
    q_ui: float
        ... flux
    hMax: float
        maximum storage volume
    dt: float
        timestep

    Returns
    -------
    v: float
        storage at current time step.
    q_oa: float
        outgoing flux

    """
    h_p = h + (q_s + q_ui + q_no) / dt  # h_potential
    if h_p > hMax:
        h = hMax
        q_oa = (hMax - h_p) / dt
    else:
        h = h_p
        q_oa = 0.0
    return h, q_oa
