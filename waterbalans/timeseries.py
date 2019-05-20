"""This file contains the function to create timeseries used in the
waterbalance.

Auteur: R.A. Collenteur, Artesia Water
        D.A. Brakenhoff, Artesia Water

"""
import logging
import numpy as np
from hkvfewspy import Pi
from pandas import DataFrame, Series, Timedelta, Timestamp, date_range


def initialize_fews_pi(wsdl='http://localhost:8080/FewsWebServices/fewspiservice?wsdl'):
    """
    FEWS Webservice 2017.01: http://localhost:8081/FewsPiService/fewspiservice?wsdl
    FEWS Webservice 2017.02: http://localhost:8080/FewsWebServices/fewspiservice?wsdl
    """
    pi = Pi()
    pi.setClient(wsdl=wsdl)
    return pi


def get_series(name, kind, data, tmin=None, tmax=None, freq="D", loggername=None):
    """Method that return a time series downloaded from fews or constructed
    from its parameters.

    Parameters
    ----------
    name: str
        Name of the returned series.
    kind: str
        String used to determine the kind of time series. Options are:
        "FEWS", "Constant" and "ValueSeries".
    data: pandas.DataFrame
    tmin: str or pandas.Timestamp, optional
        str or pandas Timestamp with the end date. Default is '2010-01-01
        00:00:00'
    tmax: str or pandas.Timestamp, optional
        str or pandas Timestamp with the end date. Default is today.
    freq: str, optional
        string with the desired frequency, not really supported now. Default
        is "D" (Daily).

    Returns
    -------
    series: pandas.Series

    Notes
    -----

    """
    if loggername is None:
        logger = logging.getLogger('waterbalans.eag')
    else:
        logger = logging.getLogger(loggername)

    try:
        pi = initialize_fews_pi()
    except Exception:
        logger.warning(
            "Pi service cannot be started. Module will not import series from FEWS!")
        pi = None

    if tmin is None:
        tmin = Timestamp("2010")
    else:
        tmin = Timestamp(tmin)
    if tmax is None:
        tmax = Timestamp.today()
    else:
        tmax = Timestamp(tmax)
    # Download a timeseries from FEWS
    if kind == "FEWS" and pi is not None:  # pragma: no cover

        # Note: this selects only the first entry if there are multiple
        if isinstance(data, DataFrame):
            data = data.loc[:, "WaardeAlfa"]
            if len(data) > 1:
                logger.warning("Multiple series found, selecting "
                               "first one ({}) and continuing".format(data.iloc[0]))
            data = data.iloc[0]
        else:
            data = data.loc["WaardeAlfa"].values[0]

        # TODO: clean this try/except stuff up
        try:
            _, moduleInstanceId, locationId, parameterId = data.split(
                "|")  # new FEWS Code
        except ValueError as e:
            try:
                moduleInstanceId, locationId, parameterId = data.split("|")
            except Exception as e:
                logger.error(
                    "Cannot parse FEWS Id for timeseries '{}'!".format(name))
                return

        query = pi.setQueryParameters(prefill_defaults=True)
        query.query["onlyManualEdits"] = False
        query.moduleInstanceIds([moduleInstanceId])
        query.locationIds([locationId])
        query.parameterIds([parameterId])
        query.startTime(tmin)
        query.endTime(tmax + Timedelta(days=1))  # add 1 day for prec/evap
        # necessary for precip data after 2016-11-30...
        query.useDisplayUnits(False)
        query.clientTimeZone('Europe/Amsterdam')

        try:
            df = pi.getTimeSeries(query, setFormat='df')
        except Exception as e:
            logger.error("FEWS Timeseries '{}': {}".format(name, e))
            return
        df.reset_index(inplace=True)
        series = df.loc[:, ["date", "value"]].set_index("date")
        series = series.tz_localize(None)  # Remove timezone from FEWS series
        series = series.astype(float)

        # omdat neerslag tussen 1jan 9u en 2jan 9u op 1jan gezet moet worden.
        if np.any(series.index.hour != 9):
            series.index = series.index.floor(freq="D") - Timedelta(days=1)
        series = series.squeeze()

        # Delete nan-values (-999) (could be moved to fewspy)
        series.replace(-999.0, np.nan, inplace=True)

        # check units, TODO, check if others need to be fixed?
        if name in ["Verdamping", "Neerslag"]:
            series = series.divide(1e3)

    # if KNMI data is required:
    elif kind == "KNMI":
        try:
            from pastas.read import KnmiStation
        except ModuleNotFoundError as e:
            logger.exception("Module 'pastas' not installed! Please intall using "
                             "pip to automatically donwload KNMI data!")
            raise e
        stn = int(data.loc[:, "Waarde"].iloc[0])
        logger.info("Downloading {0} from KNMI for station {1}...".format(
            name, stn))
        if name == "Neerslag":
            s = KnmiStation.download(
                stns=[stn], start=tmin, end=tmax, vars="RD")
            series = s.data.loc[:, "RD"]
            if np.any(series.index.hour == 9):
                series.index = series.index.floor(freq="D") - Timedelta(days=1)
            elif np.any(series.index.hour == 1):
                series.index = series.index.normalize() - Timedelta(days=1)
        elif name == "Verdamping":
            s = KnmiStation.download(
                stns=[stn], start=tmin, end=tmax, vars="EV24")
            series = s.data.loc[:, "EV24"]
            if np.any(series.index.hour == 1):
                series.index = series.index.normalize() - Timedelta(days=1)
        logger.info("Success!")

    #  If a constant timeseries is required
    elif kind == "Constant":
        if name in ["Qkwel", "Qwegz"]:
            value = float(data.loc[:, "Waarde"].values[0]) * 1e-3
        else:
            value = float(data.loc[:, "Waarde"].values[0])
        tindex = date_range(tmin, tmax, freq=freq)
        series = Series(value, index=tindex)

    # If a alternating time series is required (e.g. summer/winter level)
    elif kind == "ValueSeries":
        df = data.loc[:, ["StartDag", "Waarde"]].set_index("StartDag")
        tindex = date_range(tmin, tmax, freq=freq)
        series = create_block_series(df, tindex)
        if name in ["Qkwel", "Qwegz"]:
            series = series * 1e-3  # TODO: is this always true?

    # elif kind == "Local":
        # if kind is Local, read Series from CSV provided by dbase!
        # TODO: intuitive method to read CSV/Series from Database
        # series = pd.read_csv(data.loc[])
        # series = series[name]
        # pass

    else:
        # TODO: fix logging, commented out now, because too much noise.
        logger.warning(
            "Adding series '{0}' of kind '{1}' not supported.".format(name, kind))
        return

    series.name = name

    return series


def create_block_series(data, tindex):
    """Method that returns a series with alternating values, for example a
    summer and winter level.

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame containing the columns "StartDag" and "Waarde". Where the
        starting day (StartDag) is structured as dd-mm (01-12, 1st of
        december).
    tindex: pandas.DatetimeIndex
        Datetimeindex to use as the index of the series that are returned.

    Returns
    -------
    series: pandas.Series
        The constructed block series

    """
    # start value series 1 year before given index to ensure first period is also filled correctly.
    series = Series(index=date_range(
        tindex[0]-Timedelta(days=365), tindex[-1]))
    for t, val in data.iterrows():
        day, month = [int(x) for x in t.split("-")]
        mask = (series.index.month == month) & (series.index.day == day)
        series.loc[mask] = float(val.values[0])

    series.fillna(method="ffill", inplace=True)
    return series.loc[tindex]


def update_series(series_orig, series_new, method="append"):
    """Update timeseries with new timeseries. Either append data or overwrite
    old series wherever new series has data. Assumes daily timesteps. Will not
    work with non-daily timestamps.

    Parameters
    ----------
    series_orig : pandas.Series
        Original series
    series_new : pandas.Series
        Series used to update original series
    method : str, optional
        update method (the default is "append", which adds all data from new
        series after last entry in original). Method "overwrite", overwrites
        all data in original series where new series contains data

    Returns
    -------
    pandas.Series
        updated series

    """
    series_new = series_new.dropna()
    series_orig = series_orig.dropna()

    tmin = np.min([series_orig.index[0], series_new.index[0]])
    tmax = np.max([series_orig.index[-1], series_new.index[-1]])

    updated_series = Series(index=date_range(tmin, tmax, freq="D"))
    updated_series.loc[series_orig.index] = series_orig

    if method == "append":
        append_series = series_new.loc[series_orig.index[0]:tmax]
        shared_index = updated_series.index.intersection(append_series.index)
        updated_series.loc[shared_index] = append_series
    elif method == "overwrite":
        shared_index = updated_series.index.intersection(series_new.index)
        updated_series.loc[shared_index] = series_new.loc[shared_index]
    else:
        raise NotImplementedError("Method {} not implemented!".format(method))

    return updated_series
