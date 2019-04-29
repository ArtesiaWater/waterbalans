from .buckets import Bucket
from .eag import Eag
from .gaf import Gaf
from .water import Water


def create_eag(id, name, buckets, gaf=None, series=None, use_waterlevel_series=False):
    """Method to create an instance of EAG.

    Parameters
    ----------
    id: int
        integer id of the EAG.
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
    eag = Eag(id=id, name=name, gaf=gaf, series=series)

    # Voeg bakjes toe
    for _, bucket in buckets.iterrows():
        kind = bucket.loc["BakjePyCode"]
        id = bucket.loc["BakjeID"]
        area = bucket.loc["OppWaarde"]
        if kind == "Water":
            Water(id=id, eag=eag, series=series, area=area,
                  use_waterlevel_series=use_waterlevel_series)
        else:
            Bucket(kind=kind, eag=eag, id=id, area=area, series=None)

    return eag


def create_gaf(id, name, gafbuckets=None, eags=[], series=None,
               use_waterlevel_series=False):
    """Create instance of a GAF.

    Parameters
    ----------
    id : int
        integer id of the Gaf
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
    gaf = Gaf(id=id, name=name, series=series)

    # if Gaf has not been split into EAGs use gafbucket df as model structure
    if gafbuckets is not None:
        e = create_eag(gafbuckets.loc[0, "EAGID"], gaf.name,
                       gafbuckets, gaf=gaf, series=series,
                       use_waterlevel_series=use_waterlevel_series)
        gaf.add_eag(e)
    else:  # add eags if they are provided
        for e in eags:
            gaf.add_eag(e)
    return gaf
