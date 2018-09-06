import pandas as pd

import waterbalans as wb

# %% Maak een model automatisch aan
buckets = pd.read_csv("data\\opp_2019_2501-EAG-1.csv", delimiter=";",
                      decimal=",")
name = "2501-EAG-01"
id = 1
e = wb.create_eag(id, name, buckets)

reeksen = pd.read_csv("data\\reeks_2019_2501-EAG-1.csv", delimiter=";",
                      decimal=",")

for id, df in reeksen.groupby(["BakjeID", "ParamCode", "ParamType"]):
    BakjeID, ParamCode, ParamType = id
    series = wb.get_series(ParamCode, ParamType, df, "1995", "2015", "D")
    if BakjeID in e.buckets.keys():
        e.buckets[BakjeID].series[ParamCode] = series
    elif BakjeID == e.water.id:
        e.water.series[ParamCode] = series
    elif BakjeID == -9999:
        e.series[ParamCode] = series
    series.plot()

params = pd.read_csv("data\\param_2019_2501-EAG-1.csv", delimiter=";",
                     decimal=",")
e.simulate(params=params, tmin="1996")

# Calculate and plot the fluxes as a bar plot
fluxes = e.aggregate_fluxes()
fluxes.loc["2010": "2015"].resample("M").mean().plot.bar(stacked=True)

# Calculate and plot the
C = e.calculate_chloride_concentration()
C.plot()


