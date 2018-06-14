"""Dit bestand bevat het basis voorbeeld voor het gebruik van de waterbalans
module.

"""

import pandas as pd

import waterbalans as wb
from waterbalans.utils import excel2datetime, to_summer_winter

# Import some test time series
data = pd.read_csv("data\\2501_01_reeksen.csv", index_col="Date",
                   infer_datetime_format=True, dayfirst=True)
data.index = excel2datetime(data.index, freq="D")

# Create a polder instance
series = pd.DataFrame()
series[["p", "e"]] = data.loc[:, ["Abcoude", "Schiphol (V)"]] * 1e-3
e = wb.Eag(id=250101, name="2501-EAG-1", series=series)

# Water
series = pd.DataFrame()
series["s"] = pd.Series(0.0508394 * 1e-3, index=e.series.index)
series["w"] = pd.Series(-0.1190455 * 1e-3, index=e.series.index)
series["x"] = to_summer_winter(6000.0 / 502900, 3000.0 / 502900, "04-01",
                               "10-01", e.series.index)
b = wb.Water(id=1, eag=e, series=series, area=502900)

# Verhard
series = pd.DataFrame()
series["s"] = pd.Series(0.0 * 1e-3, e.series.index)
b = wb.Verhard(id=2, eag=e, series=series, area=102894)

# Gedraineerd
# series = pd.DataFrame()
# series["s"] = pd.Series(0.0, e.series.index)
# b = wb.Drain(id=3, eag=e, series=series, area=0)

# Onverhard: >0 & <= 0.5 kwel
series = pd.DataFrame()
series["s"] = pd.Series(0.087185 * 1e-3, e.series.index)
b = wb.Onverhard(id=4, eag=e, series=series, area=1840576)

# Onverhard: >-0.4306 & <= 0 weinig wegzijging
series = pd.DataFrame()
series["s"] = to_summer_winter(-0.122533 * 1e-3, -0.163377 * 1e-3, "04-01",
                               "10-01", e.series.index)
b = wb.Onverhard(id=5, eag=e, series=series, area=766779)

# Onverhard: >-1 & <=-0.4306 meer wegzijging
series = pd.DataFrame()
series["s"] = to_summer_winter(-0.534395 * 1e-3, -0.712527 * 1e-3, "04-01",
                               "10-01", e.series.index)
b = wb.Onverhard(id=6, eag=e, series=series, area=520862)

#%% Bereken de waterbalans en plot
params = pd.read_excel("data\\2501_01_parameters.xlsx")
e.simulate(params=params)

# Calculate and plot the fluxes as a bar plot
fluxes = e.aggregate_fluxes()
fluxes.loc["2010":"2015"].resample("M").mean().plot.bar(stacked=True)

# Calculate and plot the
C = e.calculate_chloride_concentration()
C.plot()
