from .buckets import Bucket
from .eag import Eag, logging
from .gaf import Gaf
from .water import Water


def create_eag(
    idn,
    name,
    buckets,
    gaf=None,
    series=None,
    use_waterlevel_series=False,
    logfile=None,
    log_level=logging.INFO,
):
    """Method to create an instance of EAG.

    Parameters
    ----------
    idn: int
        integer ID of the EAG.
    name: str
        string with the name of the EAG.
    buckets: pandas.DataFrame
        DataFrame containing the description of the buckets that need to be
        added to the Eag model.
    series: pandas.DataFrame, optional
        DataFrame with the timeseries necessary for simulation of the water
        balance.

    Returns
    -------
    eag: waterbalans.Eag instance
        Instance of the Eag class.

    Notes
    -----
    """
    eag = Eag(
        idn=idn, name=name, gaf=gaf, series=series, logfile=logfile, log_level=log_level
    )
    eag.logger.info("Creating EAG object for '{}'".format(name))

    # Voeg bakjes toe
    for _, bucket in buckets.iterrows():
        kind = bucket.loc["BakjePyCode"]
        idn = bucket.loc["BakjeID"]
        area = bucket.loc["OppWaarde"]
        if kind == "Water":
            Water(
                idn=idn,
                eag=eag,
                series=series,
                area=area,
                use_waterlevel_series=use_waterlevel_series,
            )
        else:
            Bucket(kind=kind, eag=eag, idn=idn, area=area, series=None)

    return eag


def create_gaf(
    idn, name, gafbuckets=None, eags=None, series=None, use_waterlevel_series=False
):
    """Create instance of a GAF.

    Parameters
    ----------
    idn : int
        integer ID of the Gaf
    name : str
        name of Gaf
    gafbuckets : pd.DataFrame, optional
        pandas.DataFrame containing modelstructure of the Gaf,
        use if the Gaf has not been split into subunits (Eags)
        (the default is None, which makes function look in
        eags). If provided ignores eags kwarg.
    eags : list, optional
        list of Eags that are located in the Gaf (the default
        is an empty list, which means there are no EAGs in GAF).
        Only read if gafbuckets is not provided.
    series : pandas.DataFrame, optional
        DataFrame with the timeseries necessary for simulation of the water
        balance.
    use_waterlevel_series : bool, optional
        setting whether to calculate water balance with measured
        water level series as (the default is False)

    Returns
    -------
    gaf: waterbalance.Gaf
        instance of Gaf object
    """
    gaf = Gaf(idn=idn, name=name, series=series)
    gaf.logger.info("Creating GAF object for '{}':".format(name))

    # if Gaf has not been split into EAGs use gafbucket df as model structure
    if gafbuckets is not None:
        e = create_eag(
            gafbuckets.loc[0, "EAGID"],
            gaf.name,
            gafbuckets,
            gaf=gaf,
            series=series,
            use_waterlevel_series=use_waterlevel_series,
        )
        gaf.add_eag(e)
    elif eags is not None:  # add eags if they are provided
        for e in eags:
            gaf.add_eag(e)
    return gaf
