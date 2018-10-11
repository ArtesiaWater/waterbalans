"""

Dit voorbeeld bevat de automatische simulatie van een waterbalans op
EAG-niveau. De volgende drie invoerbestanden worden gebruikt:

- Modelstructuur
- Tijdreeksen
- Parameters

"""

import pandas as pd

import waterbalans as wb

# Import some test time series
# data = pd.read_csv("data\\2501_01_reeksen.csv", index_col="Date",
#                    infer_datetime_format=True, dayfirst=True)
# data.index = wb.excel2datetime(data.index, freq="D")
# series = pd.DataFrame()
# series[["Neerslag", "Verdamping"]] = \
#     data.loc[:"2015-12-31", ["Abcoude", "Schiphol (V)"]] * 1e-3


buckets = pd.read_csv("data\\opp_16088_2501-EAG-1.csv", delimiter=";",
                      decimal=",")
name = "2501-EAG-01"
id = 1
e = wb.create_eag(id, name, buckets)

# Lees de tijdreeksen in
reeksen = pd.read_csv("data\\reeks_16088_2501-EAG-1.csv", delimiter=";",
                      decimal=",")
for id, df in reeksen.groupby(["BakjeID", "ClusterType", "ParamType"]):
    BakjeID, ClusterType, ParamType = id
    series = wb.get_series(ClusterType, ParamType, df, "1995", "2015-12-31", "D")
    if BakjeID in e.buckets.keys():
        e.buckets[BakjeID].series[ClusterType] = series
    elif BakjeID == e.water.id:
        e.water.series[ClusterType] = series
    elif BakjeID == -9999:
        e.series[ClusterType] = series

# Simuleer de waterbalans
params = pd.read_csv("data\\param_16088_2501-EAG-1.csv", delimiter=";",
                     decimal=",")

e.simulate(params=params, tmin="1996", tmax="2015-12-31")

# Calculate and plot the fluxes as a bar plot
fluxes = e.aggregate_fluxes()
fluxes.loc["2010": "2015"].resample("M").mean().plot.bar(stacked=True)

# Calculate and plot the
C = e.calculate_chloride_concentration()
C.plot()
