"""Dit bestand bevat het basis voorbeeld voor het gebruik van de waterbalans
module.

"""

import pandas as pd

import waterbalans as wb
from waterbalans.utils import excel2datetime, to_summer_winter

# Import some test time series
data = pd.read_csv("data\\2501_reeksen.csv", index_col="Date",
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
series = pd.DataFrame()
series["s"] = pd.Series(0.0, e.series.index)
b = wb.Drain(id=3, eag=e, series=series, area=0)

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

# Onverhard: <=-1.26 veel wegzijging
series = pd.DataFrame()
series["s"] = to_summer_winter(-1.160258 * 1e-3, -1.547010 * 1e-3, "04-01",
                               "10-01", e.series.index)
b = wb.Onverhard(id=7, eag=e, series=series, area=0)

params = pd.read_excel("data\\2501_01_parameters.xlsx")
e.simulate(params=params)

fluxes = e.aggregate_fluxes()
#
fluxes.loc["2010":"2015"].resample("M").mean().plot.bar(stacked=True)

# Bereken de chloride concentratie
C_init = 90
V_init = 170986
Mass = pd.Series(index=fluxes.index)
M = C_init * V_init
C_out = C_init

# Som van de uitgaande fluxen: wegzijging, intrek, uitlaat
V_out = fluxes.loc[:, ["intrek", "uitlaat", "wegzijging"]].sum(axis=1)

c = {
    "neerslag": 6,
    "kwel": 400,
    "verhard": 10,
    "riolering": 100,
    "drain": 70,
    "uitspoeling": 70,
    "afstroming": 35,
    "inlaat": 100
}
names = ["neerslag", "kwel", "verhard", "drain",
         "uitspoeling", "afstroming", "inlaat"]
c = [c[name] for name in names]

for t in fluxes.index:
    M_in =  fluxes.loc[t, names].multiply(c).sum()

    M_out = V_out.loc[t] * C_out

    M = M + M_in + M_out

    Mass.loc[t] = M
    C_out = M / e.water.storage.loc[t]

(Mass/e.water.storage).plot()
(Mass/e.water.storage).head()
