import pandas as pd

import waterbalans as wb
from waterbalans.utils import excel2datetime, to_summer_winter


# %% Definieer de functies voor het automatisch genereren van een Eag Model.
def load_series():
    data = pd.read_csv("data\\2501_01_reeksen.csv", index_col="Date",
                       infer_datetime_format=True, dayfirst=True)
    data.index = excel2datetime(data.index, freq="D")
    series = pd.DataFrame()
    series[["prec", "evap"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3

    return series


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

    """
    eag = wb.Eag(id=id, name=name, gaf=gaf, series=series)

    # Voeg bakjes toe
    for x, bucket in buckets.iterrows():
        kind = bucket.loc["Bakjes_PyCode"]
        id = bucket.loc["Bakjes_ID"]
        area = bucket.loc["Opp_Waarde"]
        if kind == "Water":
            wb.Water(id=id, eag=eag, series=series, area=area)
        else:
            wb.buckets.Bucket(kind=kind, eag=eag, id=id, area=area,
                              series=None)

    return eag

reeksen = pd.read_csv("data\\2501_01_reeksen_id.csv", delimiter=";",
                      decimal=",")
# TODO Hier ben ik gebleven. Aanmaken van tijdreeksen


# %% Maak een model automatisch aan
buckets = pd.read_csv("data\\2501_01_structuur.csv")
name = "2501-EAG-01"
id = 1
series = load_series()
e = create_eag(id, name, buckets, series=series)





e.buckets[6559].series["seep"] = pd.Series(0.087185 * 1e-3, series.index)
e.buckets[6560].series["seep"] = pd.Series(0.087185 * 1e-3, series.index)
e.buckets[6561].series["seep"] = to_summer_winter(-0.122533 * 1e-3, -0.163377 *
                                               1e-3, "04-01", "10-01",
                                               series.index)
e.buckets[6562].series["seep"] = to_summer_winter(-0.122533 * 1e-3, -0.163377 *
                                               1e-3, "04-01", "10-01",
                                               series.index)
e.buckets[6563].series["seep"] = to_summer_winter(-0.534395 * 1e-3,
                                                   -0.712527 *
                                               1e-3, "04-01", "10-01",
                                               series.index)
e.buckets[6564].series["seep"] = to_summer_winter(-0.534395 * 1e-3, -0.712527 *
                                               1e-3, "04-01", "10-01",
                                               series.index)

e.water.series["seep"] = pd.Series(0.0508394 * 1e-3, index=series.index)
e.water.series["w"] = pd.Series(-0.1190455 * 1e-3, index=series.index)
e.water.series["x"] = to_summer_winter(6000.0 / 502900, 3000.0 / 502900,
                                       "04-01", "10-01", series.index)

params = pd.read_csv("data\\2501_01_parameters.csv", delimiter=";", decimal=",")
e.simulate(params=params)

# Calculate and plot the fluxes as a bar plot
fluxes = e.aggregate_fluxes()
fluxes.loc["2010":"2015"].resample("M").mean().plot.bar(stacked=True)

# Calculate and plot the
C = e.calculate_chloride_concentration()
C.plot()
