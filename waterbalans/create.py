from .eag import Eag
from .water import Water
from .buckets import Bucket


def create_eag(id, name, buckets, gaf=None, series=None):
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
            Water(id=id, eag=eag, series=series, area=area)
        else:
            Bucket(kind=kind, eag=eag, id=id, area=area, series=None)

    return eag
