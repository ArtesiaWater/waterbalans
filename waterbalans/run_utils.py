import logging
import os

import pandas as pd

from .create import create_eag
from .utils import add_timeseries_to_obj, create_csvfile_table


def run_eag_by_name(
    name,
    csvdir,
    extra_iter=0,
    tmin="1996",
    tmax="2019",
    log_level=logging.INFO,
    logfile=None,
):
    file_df = create_csvfile_table(csvdir)
    fbuckets, fparams, freeks, fseries, _, _ = file_df.loc[name]

    dfdict = get_dataframes_from_files(
        fbuckets=fbuckets,
        fparams=fparams,
        freeks=freeks,
        fseries=fseries,
        csvdir=csvdir,
    )

    deelgebieden = dfdict["deelgebieden"]
    tijdreeksen = dfdict["tijdreeksen"]
    parameters = dfdict["parameters"]
    series = dfdict["series"]

    # %% Simulation settings based on parameters
    # ------------------------------------------
    if (
        parameters.loc[parameters.ParamCode == "hTargetMin", "Waarde"].iloc[0]
        != -9999.0
    ):
        use_wl = True
    else:
        use_wl = False

    # %% Model
    # --------
    # Maak bakjes model
    e = create_eag(
        name.split("-")[-1],
        name,
        deelgebieden,
        use_waterlevel_series=use_wl,
        logfile=logfile,
        log_level=log_level,
    )

    # Voeg tijdreeksen toe
    e.add_series_from_database(tijdreeksen, tmin=tmin, tmax=tmax)

    # Voeg overige tijdreeksen toe (overschrijf FEWS met Excel)
    if series is not None:
        series.drop(
            [
                icol
                for icol in series.columns
                if icol.lower().startswith("inlaatcalibratie1")
            ],
            axis=1,
            inplace=True,
        )
        add_timeseries_to_obj(e, series, overwrite=True)

    # Force MengRiool to use external timeseries
    mengriool = e.get_buckets(buckettype="MengRiool")
    if len(mengriool) > 0:
        for b in mengriool:
            b.use_eag_cso_series = False
            b.path_to_cso_series = os.path.join(
                csvdir, "../cso_series/240_cso_timeseries.pklz"
            )

    # Simuleer waterbalans met parameters
    if extra_iter == 0:
        e.simulate(parameters, tmin=tmin, tmax=tmax)
    elif extra_iter > 0:
        e.simulate_iterative(parameters, extra_iters=extra_iter, tmin=tmin, tmax=tmax)

    return e


def get_dataframes_from_files(
    csvdir, fbuckets=None, freeks=None, fparams=None, fseries=None
):
    dflist = {}
    if fbuckets is not None:
        # bestand met deelgebieden en oppervlaktes:
        deelgebieden = pd.read_csv(os.path.join(csvdir, fbuckets), delimiter=";")
        dflist["deelgebieden"] = deelgebieden
    if freeks is not None:
        # bestand met tijdreeksen, b.v. neerslag/verdamping:
        tijdreeksen = pd.read_csv(os.path.join(csvdir, freeks), delimiter=";")
        dflist["tijdreeksen"] = tijdreeksen
    if fparams is not None:
        # bestand met parameters per deelgebied
        parameters = pd.read_csv(os.path.join(csvdir, fparams), delimiter=";")
        dflist["parameters"] = parameters
    if fseries is not None:
        # bestand met overige tijdreeksen
        if not isinstance(fseries, float):
            series = pd.read_csv(
                os.path.join(csvdir, fseries),
                delimiter=";",
                index_col=[0],
                parse_dates=True,
                dayfirst=True,
            )
        else:
            series = None
        dflist["series"] = series
    return dflist


def get_dataframes_by_name(name, csvdir):
    file_df = create_csvfile_table(csvdir)
    fbuckets, fparams, freeks, fseries, _, _ = file_df.loc[name]

    dfdict = get_dataframes_from_files(
        csvdir=csvdir,
        fbuckets=fbuckets,
        fparams=fparams,
        freeks=freeks,
        fseries=fseries,
    )
    return dfdict
